"""Time-domain haul-cycle simulator.

Drives a truck through a scenario (load, loaded leg, dump, empty leg) with a chosen driver
style and produces the 1 Hz telemetry a real machine would publish, plus hidden ground
truth about unsafe behaviour and faults for evaluating the detectors.

The physics is the same ``Vehicle`` model the planner uses. Unlike the planner, the driver
here is imperfect: it reacts to limits with a style-dependent look-ahead, wanders in speed,
shifts by rpm thresholds, meets hazards typical of mixed Indian traffic (cattle, tractors,
two-wheelers cutting in), and may coast downhill in neutral.
"""

from __future__ import annotations

import math
from collections.abc import Iterator
from dataclasses import dataclass, field

import numpy as np

from .constants import DRIVE_AXLE_LOAD_SHARE, G
from .drivers import (
    EPISODE_KINDS,
    EPISODE_WEIGHTS,
    DriverProfile,
    Episode,
    FaultSpec,
    get_profile,
)
from .engine import RPM_IDLE, engine_brake_torque, full_load_torque
from .planner import Plan, PlannerConfig, plan_leg
from .routes import Route, RouteProfile, Scenario, get_scenario
from .schemas import Telemetry
from .vehicle import Vehicle

_PLAN_CACHE: dict = {}

SHIFT_TIME_S = 0.5  # torque interruption during a gear change


@dataclass(frozen=True)
class Truth:
    """Hidden ground truth for one telemetry sample. Never published."""

    unsafe: bool
    episode: str | None
    neutral_coast: bool
    near_miss: bool
    fault: str | None


@dataclass
class SimResult:
    telemetry: list[Telemetry] = field(default_factory=list)
    truth: list[Truth] = field(default_factory=list)

    @property
    def n(self) -> int:
        return len(self.telemetry)


class Simulation:
    def __init__(
        self,
        scenario: str | Scenario,
        driver: str | DriverProfile = "typical",
        *,
        vehicle: Vehicle | None = None,
        machine_id: str = "SIM-001",
        seed: int = 0,
        trips: int = 1,
        faults: tuple[FaultSpec, ...] = (),
        ambient_c: float = 34.0,
        t0_wall: float = 1_700_000_000.0,
        initial_fuel_l: float = 240.0,
        payload_kg: float | None = None,
        plan_config: PlannerConfig | None = None,
        dt: float = 0.5,
        start_trip_no: int = 1,
        hazards: bool = True,
    ):
        self.scn = get_scenario(scenario) if isinstance(scenario, str) else scenario
        self.driver = get_profile(driver)
        self.veh = vehicle or Vehicle()
        self.machine_id = machine_id
        self.seed = seed
        self.hazards_enabled = hazards
        self.rng = np.random.default_rng(seed)
        self.noise = np.random.default_rng(seed + 7919)
        self.trips = trips
        self.faults = tuple(faults)
        self.ambient = ambient_c
        self.t0_wall = t0_wall
        self.fixed_payload_kg = payload_kg
        self.plan_config = plan_config or PlannerConfig()
        self.dt = dt
        self.trip_no = start_trip_no - 1

        # vehicle state
        self.t = 0.0
        self.v = 0.0
        self.pos = 0.0
        self.gear = 1
        self.pending_gear = 1
        self.shift_left = 0.0
        self.last_shift_t = -10.0
        self.engine_on = True
        self.payload_kg = 0.0
        self.leg = "loaded"
        self.route_id = self.scn.id
        self.prof: RouteProfile | None = None
        self.plan: Plan | None = None
        self.grade = 0.0
        self.speed_limit_ms = 0.0

        # counters
        self.fuel_total_l = 0.0
        self.odo_m = 0.0
        self.engine_s = 0.0
        self.idle_s = 0.0
        self.idle_fuel_l = 0.0
        self.fuel_level_l = initial_fuel_l

        # thermal and sensors
        self.coolant = 80.0
        self.brake_temp = ambient_c + 15.0
        self.rpm = RPM_IDLE
        self._theft_left = 0.0
        self._theft_done = False

        # driver state
        self._ou = 0.0
        self._braking = False
        self._haz_left = 0.0
        self._haz_v = 0.0
        self._haz_a = 0.0
        self._haz_events: list[tuple[float, float, float]] = []
        self._haz_idx = 0
        self._near_miss = False
        self._ep: Episode | None = None
        self._next_ep_t = self._draw_gap()
        self.coast_desc = False
        self._prev_grade = 0.0
        self._wait_s = 0.0

        # per-second accumulators
        self._acc_fuel_l = 0.0
        self._acc_a = 0.0
        self._acc_n = 0
        self._last_a = 0.0
        self._last_throttle = 0.0
        self._last_brake = 0.0
        self._last_torque_pct = 0.0
        self._ticks_per_s = max(1, round(1.0 / dt))
        self._tick = 0

    # ------------------------------------------------------------------ public API
    def stream(self) -> Iterator[tuple[Telemetry, Truth]]:
        """Yield (telemetry, truth) at 1 Hz for every requested trip."""
        sc = self.scn
        for _ in range(self.trips):
            self.trip_no += 1
            payload = self._draw_payload()
            self._maybe_refuel()
            yield from self._dwell(sc.load_dwell_s, sc.loaded, "loaded", 0.0, 0.0, payload, shutdown=True)
            yield from self._drive_leg("loaded", sc.loaded, payload)
            yield from self._dwell(sc.dump_dwell_s, sc.loaded, "loaded", sc.loaded.length_m, payload, 0.0, shutdown=True)
            yield from self._drive_leg("empty", sc.empty, 0.0)

    def run(self) -> SimResult:
        out = SimResult()
        for tel, truth in self.stream():
            out.telemetry.append(tel)
            out.truth.append(truth)
        return out

    # ------------------------------------------------------------------ set-up helpers
    def _draw_gap(self) -> float:
        rate = self.driver.episode_rate_per_h
        return float(self.rng.exponential(3600.0 / rate)) + self.t if rate > 0 else math.inf

    def _draw_payload(self) -> float:
        if self.fixed_payload_kg is not None:
            return float(self.fixed_payload_kg)
        d = self.driver
        rated = self.veh.params.payload_rated_kg
        frac = float(self.rng.uniform(d.payload_lo, d.payload_hi))
        if d.overload_prob > 0 and self.rng.random() < d.overload_prob:
            frac = 1.0 + float(self.rng.uniform(0.05, d.overload_max))
        return frac * rated

    def _maybe_refuel(self) -> None:
        if self.fuel_level_l < 0.35 * self.veh.params.tank_l:
            self.fuel_level_l = 0.95 * self.veh.params.tank_l

    def _get_plan(self, route: Route, leg: str, payload_kg: float) -> Plan:
        key = (id(self.veh.fuel_map), self.veh.params, route.id, leg, round(payload_kg / 500.0), self.plan_config)
        if key not in _PLAN_CACHE:
            _PLAN_CACHE[key] = plan_leg(self.veh, route.profile(), payload_kg, self.plan_config)
        return _PLAN_CACHE[key]

    def _fault(self, kind: str) -> bool:
        return any(f.kind == kind and self.t >= f.start_s for f in self.faults)

    def _active_fault(self) -> str | None:
        for f in self.faults:
            if self.t >= f.start_s:
                if f.kind == "fuel_theft" and not (self._theft_left > 0):
                    continue
                return f.kind
        return None

    # ------------------------------------------------------------------ legs and dwells
    def _dwell(
        self,
        seconds: float,
        route: Route,
        leg: str,
        pos: float,
        payload_from: float,
        payload_to: float,
        *,
        shutdown: bool,
    ) -> Iterator[tuple[Telemetry, Truth]]:
        self.prof = route.profile()
        self.leg, self.route_id = leg, self.scn.id
        self.v, self.pos = 0.0, pos
        self.grade = 0.0
        self.speed_limit_ms = float(self.prof.legal_ms[min(int(pos / self.prof.ds), self.prof.n_nodes - 1)])
        n = int(round(seconds / self.dt))
        self._wait_s = 0.0
        for k in range(n):
            frac = (k + 1) / max(n, 1)
            self.payload_kg = payload_from + (payload_to - payload_from) * frac
            # eco drivers switch the engine off during long waits and restart just before leaving
            remaining = (n - k) * self.dt
            if shutdown and self.driver.idle_shutdown_s is not None:
                self.engine_on = not (self._wait_s >= self.driver.idle_shutdown_s and remaining > 3.0)
            else:
                self.engine_on = True
            self._start_theft_if_due()
            self._physics_idle()
            self._wait_s += self.dt
            self._after_tick()
            sample = self._maybe_emit()
            if sample:
                yield sample
        self.engine_on = True
        self._wait_s = 0.0
        self.gear = 1

    def _start_theft_if_due(self) -> None:
        if self._theft_done or self._theft_left > 0:
            return
        for f in self.faults:
            if f.kind == "fuel_theft" and self.t >= f.start_s and self.v < 0.1:
                self._theft_left = 60.0
                self._theft_done = True

    def _drive_leg(self, leg: str, route: Route, payload_kg: float) -> Iterator[tuple[Telemetry, Truth]]:
        prof = route.profile()
        self.prof = prof
        self.leg, self.route_id = leg, self.scn.id
        self.payload_kg = payload_kg
        self.plan = self._get_plan(route, leg, payload_kg) if self.driver.follow_plan else None
        mass = self.veh.mass(payload_kg)
        self.v, self.pos, self.gear = 0.0, 0.0, 1
        self.shift_left = 0.0
        self.coast_desc = False
        self._braking = False
        self._haz_left = 0.0
        self._haz_events = self._make_hazards(prof, leg)
        self._haz_idx = 0
        self._prev_grade = 0.0
        n_nodes = prof.n_nodes
        served = 0  # index of the last stop node already handled
        guard = int(4 * 3600 / self.dt)

        while guard > 0:
            guard -= 1
            i = prof.interval_index(self.pos)
            self.grade = float(prof.grade_pct[i])
            crr, mu = float(prof.crr[i]), float(prof.mu[i])
            self.speed_limit_ms = float(max(prof.legal_ms[i], prof.legal_ms[i + 1]))

            self._update_episode()
            self._update_coasting(i)
            a_cmd = self._control(i, mass)
            self._update_gear(a_cmd, crr, mass)
            self._physics_drive(a_cmd, crr, mu, mass)
            self._integrate_position(prof, i, served)

            self._after_tick()
            sample = self._maybe_emit()
            if sample:
                yield sample

            # arrived at a stop node?
            j = i + 1
            if self.v == 0.0 and j > served and prof.cap_ms[j] <= 0.0 and abs(self.pos - prof.s[j]) < 1e-6:
                served = j
                if j == n_nodes - 1:
                    return
                yield from self._dwell(
                    float(prof.dwell_s[j]), route, leg, self.pos, payload_kg, payload_kg, shutdown=True
                )
                self.prof = prof
                self.leg = leg
                self.gear = 1
        raise RuntimeError(f"simulation of {route.id}/{leg} did not finish")

    # ------------------------------------------------------------------ driver model
    def _update_episode(self) -> None:
        if self._ep and self.t >= self._ep.end_t:
            self._ep = None
            self._next_ep_t = self._draw_gap()
        if self._ep is None and self.t >= self._next_ep_t and self.v > 3.0:
            kind = str(self.rng.choice(EPISODE_KINDS, p=EPISODE_WEIGHTS))
            self._ep = Episode(kind, self.t, self.t + float(self.rng.uniform(25.0, 60.0)))

    def _update_coasting(self, i: int) -> None:
        d = self.driver
        g = self.grade
        if g < -2.5 and self._prev_grade >= -2.5 and self.rng.random() < d.neutral_coast_prob:
            self.coast_desc = True
        if self.coast_desc and (g > -1.0 or self.v > 1.25 * float(self.prof.cap_ms[i]) + 2.0):
            self.coast_desc = False  # panic: engage a gear, or the slope has ended
        self._prev_grade = g

    @property
    def coast_neutral(self) -> bool:
        ep_coast = self._ep is not None and self._ep.kind == "neutral_coast" and self.v > 5.0
        return bool(self.coast_desc or ep_coast)

    def _lookahead(self, i: int, cf: float) -> tuple[float, int]:
        prof = self.prof
        lo = i + 1
        hi = min(lo + 24, prof.n_nodes)
        idx = np.arange(lo, hi)
        dist = np.maximum(prof.s[idx] - self.pos, 1.0)
        vlim = prof.cap_ms[idx] * cf
        a_req = np.where(self.v > vlim, (self.v**2 - vlim**2) / (2.0 * dist), 0.0)
        k = int(a_req.argmax())
        return float(a_req[k]), int(idx[k])

    def _control(self, i: int, mass: float) -> float:
        d = self.driver
        prof = self.prof
        dt = self.dt
        v = self.v
        ep = self._ep.kind if self._ep else None

        kp, headway, a_look, noise_mult = d.kp, d.headway_s, d.look_decel, 1.0
        if ep == "tailgate":
            headway, a_look = 0.7, a_look * 1.5
        elif ep == "erratic":
            noise_mult, kp = 5.0, kp * 0.6

        # speed-target wander (Ornstein-Uhlenbeck), larger for sloppy drivers
        tau = 5.0
        self._ou += (-self._ou / tau) * dt + d.speed_noise * noise_mult * math.sqrt(2 * dt / tau) * self.noise.normal()

        if d.follow_plan and self.plan is not None:
            pv = self.plan.v
            k = min(i, pv.size - 2)
            a_plan = (pv[k + 1] ** 2 - pv[k] ** 2) / (2.0 * prof.ds)  # acceleration the plan implies here
            v_t = self.plan.target_speed(self.pos)
            a = float(np.clip(a_plan + kp * (v_t - v), -d.a_dec_max, d.a_acc))
            return self._hazards(a, headway, kp)

        cap_int = max(float(prof.cap_ms[i]), float(prof.cap_ms[i + 1]))
        cruise = cap_int * d.cap_factor
        if ep == "overspeed":
            legal = max(float(prof.legal_ms[i]), float(prof.legal_ms[i + 1]))
            cruise = max(cruise, min(1.2 * legal, 21.0))
        cruise = max(1.5, cruise + self._ou)

        # look-ahead braking for limits, bumps, hairpins and stops
        a_req, _ = self._lookahead(i, d.cap_factor)
        if self._braking:
            if a_req < 0.5 * a_look:
                self._braking = False
        elif a_req >= a_look:
            self._braking = True
        if self._braking:
            a_cmd = -min(max(a_req, 0.3), d.a_dec_max)
        else:
            a_cmd = float(np.clip(kp * (cruise - v), -1.0, d.a_acc))

        # unplanned hazards: cattle, tractors, two-wheelers cutting in
        return self._hazards(a_cmd, headway, kp)

    def _make_hazards(self, prof: RouteProfile, leg: str) -> list[tuple[float, float, float]]:
        """Pre-schedule unplanned slow-downs by position, independent of the driver.

        Every driver on the same seed meets the same cattle, tractors and cut-ins, so
        differences between drivers come from how they handle them, not from luck.
        """
        rng = np.random.default_rng([self.seed, self.trip_no, 1 if leg == "loaded" else 2])
        events: list[tuple[float, float, float]] = []
        if not self.hazards_enabled:
            return events
        for k in range(prof.grade_pct.size):
            lam = float(prof.hazard_per_km[k]) * prof.ds / 1000.0
            for _ in range(int(rng.poisson(lam))):
                events.append(
                    (
                        float(prof.s[k] + rng.uniform(0.0, prof.ds)),
                        float(rng.uniform(2.0, 8.0)),  # speed of the obstruction, m/s
                        float(rng.uniform(3.0, 8.0)),  # how long it lasts, s
                    )
                )
        return sorted(events)

    def _hazards(self, a_cmd: float, headway: float, kp: float) -> float:
        d = self.driver
        v = self.v
        if self._haz_left <= 0.0:
            self._near_miss = False
            if self._haz_idx < len(self._haz_events):
                pos_h, v_h, dur = self._haz_events[self._haz_idx]
                if self.pos >= pos_h:
                    self._haz_idx += 1
                    if v > v_h + 2.0:
                        sight = max(8.0, headway * v + float(self.rng.uniform(-3.0, 3.0)))
                        self._haz_v = v_h
                        self._haz_a = (v**2 - v_h**2) / (2.0 * sight)
                        self._haz_left = dur
                        self._near_miss = self._haz_a > d.a_dec_max
            return a_cmd
        self._haz_left -= self.dt
        if v > self._haz_v + 0.3:
            return -min(max(self._haz_a, 0.3), d.a_dec_max)
        return float(np.clip(kp * (self._haz_v - v), -d.a_dec_max, 0.5))

    # ------------------------------------------------------------------ gearbox
    def _update_gear(self, a_cmd: float, crr: float, mass: float) -> None:
        veh = self.veh
        d = self.driver
        ng = veh.n_gears
        v = self.v

        if self.shift_left > 0.0:
            self.shift_left -= self.dt
            if self.shift_left <= 0.0:
                self.gear = self.pending_gear
            return
        if self.coast_neutral and v > 3.0:
            self.gear = 0
            return
        if self.gear == 0:
            self.gear = self._pick_gear(v)
            return
        if v < 0.05:
            self.gear = 1
            if self.driver.follow_plan and self.plan is not None:
                launch = self.plan.gear_at(self.pos)
                self.gear = launch if launch <= 2 else 1  # only launch gears may slip the clutch
            return

        g = self.gear
        desired = g
        if g <= 2 and v < veh.params.launch_speed_ms:
            return  # no gear changes while the clutch is still slipping at launch
        rpm = float(veh.rpm_at(v, g))
        f_need = float(veh.m_eff(g, mass)) * a_cmd + float(veh.resistance_n(v, mass, self.grade, crr))

        def usable(gear: int) -> bool:
            r = float(veh.rpm_at(v, gear))
            t = float(veh.force_to_torque(f_need, gear))
            return 900.0 <= r <= 1950.0 and t <= 0.97 * float(full_load_torque(r))

        planned = None
        if d.follow_plan and self.plan is not None:
            cand = self.plan.gear_at(self.pos)
            if usable(cand):
                planned = cand
        if planned is not None:
            desired = planned
        elif self.grade < -2.5 and a_cmd <= 0.3 and v > 3.0:
            gs = np.arange(1, ng + 1)
            ok = veh.rpm_at(v, gs) <= d.descent_rpm
            desired = int(gs[ok].min()) if ok.any() else ng
        else:
            tfull = float(full_load_torque(max(rpm, 800.0)))
            t_dem = float(veh.force_to_torque(f_need, g))
            if g < ng:
                rpm_n = rpm * float(veh.ratio(g + 1)) / float(veh.ratio(g))
                t_next = float(veh.force_to_torque(f_need, g + 1))
                if (
                    rpm > d.shift_up_rpm
                    and rpm_n >= d.shift_down_rpm + 100.0
                    and t_next <= 0.9 * float(full_load_torque(rpm_n))
                ):
                    desired = g + 1
                elif rpm > 1950.0 and rpm_n >= 800.0:
                    desired = g + 1  # running out of revs: shift whatever the torque margin
            if g > 1 and desired == g and (rpm < d.shift_down_rpm or t_dem > d.lug_frac * tfull):
                # Drivers skip gears when they have dropped well below the power band: take
                # the highest lower gear that has torque in hand and stays under the rev ceiling.
                desired = g - 1
                for cand in range(g - 1, 0, -1):
                    r = float(veh.rpm_at(v, cand))
                    if r > 1950.0:
                        break
                    tq = float(veh.force_to_torque(f_need, cand))
                    if r >= d.shift_down_rpm + 100.0 and tq <= 0.9 * d.lug_frac * float(full_load_torque(r)):
                        desired = cand
                        break

        if desired != g and (self.t - self.last_shift_t) >= 3.0:
            self.pending_gear = int(desired)
            self.shift_left = SHIFT_TIME_S
            self.last_shift_t = self.t
            self.gear = 0

    def _pick_gear(self, v: float) -> int:
        veh = self.veh
        d = self.driver
        gs = np.arange(1, veh.n_gears + 1)
        rpm = veh.rpm_at(max(v, 0.3), gs)
        ok = (rpm >= d.shift_down_rpm + 150.0) & (rpm <= d.shift_up_rpm)
        if ok.any():
            return int(gs[ok].max())
        return int(gs[np.argmin(np.abs(rpm - 0.5 * (d.shift_up_rpm + d.shift_down_rpm)))])

    # ------------------------------------------------------------------ physics
    def _physics_drive(self, a_cmd: float, crr: float, mu: float, mass: float) -> None:
        veh = self.veh
        p = veh.params
        v = self.v
        dt = self.dt
        gear = self.gear
        f_res = float(veh.resistance_n(v, mass, self.grade, crr))
        f_service_max = mu * mass * G * 0.9
        f_trac_max = mu * mass * G * DRIVE_AXLE_LOAD_SHARE

        if gear == 0:
            meff = mass + p.wheel_inertia_kgm2 / p.wheel_radius_m**2
            f_service = float(np.clip(-f_res - meff * a_cmd, 0.0, f_service_max))
            f_eng = 0.0
            rpm = RPM_IDLE
            torque = 0.0
            fuel_gps = veh.idle_fuel_gps if self.engine_on else 0.0
            tfull = float(full_load_torque(rpm))
        else:
            meff = float(veh.m_eff(gear, mass))
            rpm_w = float(veh.rpm_at(v, gear))
            slip = gear <= 2 and v < p.launch_speed_ms
            rpm = max(rpm_w, 800.0) if slip else rpm_w
            rpm = float(np.clip(rpm, RPM_IDLE, 2100.0))
            tfull = float(full_load_torque(rpm))
            f_need = meff * a_cmd + f_res
            t_req = float(veh.force_to_torque(f_need, gear))
            if rpm_w >= 2100.0:
                t_req = min(t_req, 0.0)  # governed: fuel is cut above maximum engine speed
            if t_req >= 0.0:
                torque = min(t_req, tfull)
                f_eng = min(float(veh.torque_to_force(torque, gear)), f_trac_max)
                f_service = 0.0
            else:
                torque = max(t_req, -float(engine_brake_torque(rpm)))
                f_eng = float(veh.torque_to_force(torque, gear))
                f_service = float(np.clip(f_eng - f_need, 0.0, f_service_max))
            fuel_gps = float(veh.fuel_gps(rpm, torque)) if self.engine_on else 0.0

        if self._fault("injector"):
            fuel_gps *= 1.25

        a = (f_eng - f_res - f_service) / meff
        v_new = max(0.0, v + a * dt)
        a_eff = (v_new - v) / dt
        self.pos_step = 0.5 * (v + v_new) * dt
        self.v_new = v_new

        self.rpm = rpm if self.engine_on else 0.0
        fuel_l = float(veh.grams_to_litres(fuel_gps * dt))
        self._record(fuel_l, a_eff, torque, tfull, f_service, f_service_max, f_service * v)

    def _physics_idle(self) -> None:
        """Truck stationary: idle fuel or engine off."""
        veh = self.veh
        fuel_gps = veh.idle_fuel_gps if self.engine_on else 0.0
        if self._fault("injector"):
            fuel_gps *= 1.25
        self.rpm = RPM_IDLE if self.engine_on else 0.0
        self.gear = 1
        fuel_l = float(veh.grams_to_litres(fuel_gps * self.dt))
        self.pos_step = 0.0
        self.v_new = 0.0
        self._record(fuel_l, 0.0, 0.0, float(full_load_torque(RPM_IDLE)), 0.0, 1.0, 0.0)

    def _integrate_position(self, prof: RouteProfile, i: int, served: int) -> None:
        j = i + 1
        pos_new = self.pos + self.pos_step
        v_new = self.v_new
        stop_ahead = prof.cap_ms[j] <= 0.0 and j > served
        if stop_ahead:
            d_stop = prof.s[j] - pos_new
            if d_stop <= 0.8 or (v_new < 1.0 and d_stop < 2.0) or pos_new >= prof.s[j]:
                pos_new = float(prof.s[j])
                v_new = 0.0
                self._braking = False
        self.pos = pos_new
        self.v = v_new
        self.odo_m += self.pos_step

    def _record(self, fuel_l, a, torque, tfull, f_service, f_service_max, p_brake_w) -> None:
        dt = self.dt
        self._last_a = a
        self._acc_a += a
        self._acc_n += 1
        self._acc_fuel_l += fuel_l
        self.fuel_total_l += fuel_l
        leak = 0.0
        if self._fault("fuel_leak"):
            leak += 0.6 / 60.0 * dt
        if self._theft_left > 0:
            leak += 0.5 * dt
            self._theft_left -= dt
        self.fuel_level_l = max(0.0, self.fuel_level_l - fuel_l - leak)
        if self.engine_on:
            self.engine_s += dt
            if self.v < 0.3:
                self.idle_s += dt
                self.idle_fuel_l += fuel_l
        self._last_torque_pct = 100.0 * max(torque, 0.0) / max(tfull, 1.0)
        self._last_throttle = self._last_torque_pct
        self._last_brake = 100.0 * min(f_service / max(f_service_max, 1.0), 1.0)
        p_frac = max(torque, 0.0) * self.rpm * math.pi / 30.0 / 259e3 if self.engine_on else 0.0
        self._thermal(min(p_frac, 1.0), p_brake_w)

    def _thermal(self, p_frac: float, p_brake_w: float) -> None:
        dt = self.dt
        amb = self.ambient
        if self.engine_on:
            t_ss = 82.0 + 14.0 * p_frac + 0.3 * (amb - 30.0)
            tau = 70.0
            if self._fault("overheat"):
                t_ss += 38.0
                tau = 90.0
        else:
            t_ss, tau = amb, 600.0
        self.coolant += (t_ss - self.coolant) * dt / tau
        h = 120.0 + 25.0 * self.v
        self.brake_temp += (p_brake_w - h * (self.brake_temp - amb)) * dt / 150e3

    def _after_tick(self) -> None:
        self.t += self.dt
        self._tick += 1

    # ------------------------------------------------------------------ telemetry
    def _maybe_emit(self) -> tuple[Telemetry, Truth] | None:
        if self._tick % self._ticks_per_s != 0:
            return None
        n = self.noise
        v = self.v
        speed = max(0.0, v * 3.6 + n.normal(0.0, 0.15))
        a_mean = self._acc_a / max(self._acc_n, 1)
        fuel_lph = self._acc_fuel_l / (self._acc_n * self.dt) * 3600.0
        self._acc_a = 0.0
        self._acc_n = 0
        self._acc_fuel_l = 0.0

        oil_factor = 0.4 if self._fault("low_oil") else 1.0
        oil = (
            max(0.0, (110.0 + 0.13 * (self.rpm - 600.0)) * oil_factor + n.normal(0.0, 3.0))
            if self.engine_on
            else 0.0
        )
        legal_kmh = self.speed_limit_ms * 3.6
        tel = Telemetry(
            machine_id=self.machine_id,
            ts=self.t0_wall + self.t,
            sim_t=self.t,
            trip_no=self.trip_no,
            route_id=self.route_id,
            leg=self.leg,
            pos_m=min(max(self.pos, 0.0), 500_000.0),
            speed_kmh=min(speed, 200.0),
            accel_ms2=float(np.clip(a_mean + n.normal(0.0, 0.04), -20, 20)),
            gear=int(self.gear),
            rpm=float(max(0.0, self.rpm + (n.normal(0.0, 4.0) if self.engine_on else 0.0))),
            torque_pct=float(np.clip(self._last_torque_pct, 0, 150)),
            throttle_pct=float(np.clip(self._last_throttle, 0, 100)),
            brake_pct=float(np.clip(self._last_brake, 0, 100)),
            grade_pct=float(np.clip(self.grade, -40, 40)),
            speed_limit_kmh=float(np.clip(legal_kmh, 0, 200)),
            payload_t=float(max(0.0, self.payload_kg / 1000.0 + n.normal(0.0, 0.03))),
            fuel_lph=float(max(0.0, fuel_lph * (1.0 + n.normal(0.0, 0.01)))),
            fuel_total_l=self.fuel_total_l,
            fuel_level_l=float(max(0.0, self.fuel_level_l + n.normal(0.0, 0.3))),
            odometer_km=self.odo_m / 1000.0,
            engine_s=self.engine_s,
            idle_s=self.idle_s,
            idle_fuel_l=self.idle_fuel_l,
            coolant_c=float(self.coolant + n.normal(0.0, 0.3)),
            oil_kpa=float(oil),
            brake_temp_c=float(self.brake_temp + n.normal(0.0, 1.0)),
            engine_on=self.engine_on,
        )
        ep = self._ep.kind if self._ep else None
        coast = self.coast_neutral and self.gear == 0 and v > 5.0
        truth = Truth(
            unsafe=bool(ep or coast or self._near_miss),
            episode=ep,
            neutral_coast=bool(coast),
            near_miss=bool(self._near_miss and self._haz_left > 0),
            fault=self._active_fault(),
        )
        return tel, truth
