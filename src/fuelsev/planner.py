"""Dynamic-programming speed and gear planner.

State space: (position node, speed level, gear used on the last step).
Decision: speed at the next node and gear for the next step.
Cost per step: fuel (L) + lambda x time (s) + a small penalty on service-brake energy
(so descents use the engine brake) + a gear-shift penalty.

The whole (v0, v1, gear) cost tensor for a step is computed at once with NumPy, and the
minimisation over the previous speed and gear is a handful of array operations, so a
12 km route plans in a couple of seconds.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .constants import DRIVE_AXLE_LOAD_SHARE, TIME_WEIGHT_L_PER_S, G
from .routes import RouteProfile
from .vehicle import Vehicle


class PlanInfeasible(RuntimeError):
    """No speed/gear sequence satisfies the route limits and the vehicle's capability."""


@dataclass(frozen=True)
class PlannerConfig:
    dv: float = 0.5  # speed resolution, m/s
    v_min: float = 2.0  # minimum speed while moving, m/s
    lam_l_per_s: float = TIME_WEIGHT_L_PER_S  # litres of diesel one second of time is worth
    shift_penalty_l: float = 0.02  # cost of a gear change (torque interruption, wear)
    brake_penalty_l_per_mj: float = 0.01  # discourage dumping energy in the service brakes
    a_acc_max: float = 1.0  # m/s^2
    a_dec_max: float = 1.5  # m/s^2 (comfortable braking for a loaded truck)
    rpm_run_min: float = 800.0
    rpm_run_max: float = 1950.0


@dataclass
class Plan:
    route_id: str
    payload_kg: float
    config: PlannerConfig
    s: np.ndarray  # node positions, m
    v: np.ndarray  # planned speed at each node, m/s
    gear: np.ndarray  # gear used on each interval
    dt_s: np.ndarray  # time per interval
    fuel_l: np.ndarray  # fuel per interval
    dwell_s: np.ndarray  # stop time at each node (not planned, fixed by the route)
    idle_fuel_l: float = 0.0
    objective: float = 0.0
    meta: dict = field(default_factory=dict)

    # -- totals -------------------------------------------------------------------
    @property
    def moving_fuel_l(self) -> float:
        return float(self.fuel_l.sum())

    @property
    def total_fuel_l(self) -> float:
        return self.moving_fuel_l + self.idle_fuel_l

    @property
    def moving_time_s(self) -> float:
        return float(self.dt_s.sum())

    @property
    def dwell_time_s(self) -> float:
        return float(self.dwell_s.sum())

    @property
    def distance_m(self) -> float:
        return float(self.s[-1])

    # -- lookup for coaching and the simulated eco driver ---------------------------
    def target_speed(self, pos_m: float) -> float:
        """Planned speed at a position, assuming constant acceleration within each step."""
        i = int(np.searchsorted(self.s, pos_m, side="right") - 1)
        i = min(max(i, 0), self.v.size - 2)
        step = float(self.s[i + 1] - self.s[i])
        accel = (self.v[i + 1] ** 2 - self.v[i] ** 2) / (2.0 * step)
        v2 = self.v[i] ** 2 + 2.0 * accel * (pos_m - self.s[i])
        return float(np.sqrt(max(v2, 0.0)))

    def gear_at(self, pos_m: float) -> int:
        i = int(np.searchsorted(self.s, pos_m, side="right") - 1)
        return int(self.gear[min(max(i, 0), self.gear.size - 1)])

    def summary(self) -> dict:
        return {
            "route": self.route_id,
            "payload_t": round(self.payload_kg / 1000.0, 2),
            "distance_km": round(self.distance_m / 1000.0, 2),
            "moving_time_min": round(self.moving_time_s / 60.0, 2),
            "dwell_min": round(self.dwell_time_s / 60.0, 2),
            "fuel_l": round(self.total_fuel_l, 2),
            "avg_speed_kmh": round(self.distance_m / self.moving_time_s * 3.6, 1),
            "gear_shifts": int((np.diff(self.gear) != 0).sum()),
            "objective_l": round(self.objective, 3),
        }


def plan_leg(
    vehicle: Vehicle,
    profile: RouteProfile,
    payload_kg: float,
    config: PlannerConfig | None = None,
) -> Plan:
    cfg = config or PlannerConfig()
    n = profile.n_nodes - 1
    ds = profile.ds
    mass = vehicle.mass(payload_kg)
    ng = vehicle.n_gears

    v_top = float(np.ceil(profile.cap_ms.max() / cfg.dv) * cfg.dv)
    V = np.arange(0.0, v_top + cfg.dv / 2, cfg.dv)
    nv = V.size
    gears = np.arange(1, ng + 1)
    gidx = np.arange(ng)

    # Which speed levels may be used at each node.
    allowed = np.zeros((n + 1, nv), dtype=bool)
    for k in range(n + 1):
        if profile.cap_ms[k] <= 0.0:
            allowed[k, 0] = True  # stops: speed must be zero
        else:
            allowed[k] = (V >= cfg.v_min - 1e-9) & (V <= profile.cap_ms[k] + 1e-9)

    V0 = V[:, None, None]
    V1 = V[None, :, None]
    G3 = gears[None, None, :]

    J = np.full((nv, ng), np.inf)
    J[0, :] = 0.0
    back_v = np.zeros((n, nv, ng), dtype=np.int16)
    back_g = np.zeros((n, nv, ng), dtype=np.int8)

    for i in range(n):
        tr = vehicle.transition(
            V0,
            V1,
            G3,
            ds,
            profile.grade_pct[i],
            profile.crr[i],
            mass,
            a_acc_max=cfg.a_acc_max,
            a_dec_max=cfg.a_dec_max,
            rpm_run_min=cfg.rpm_run_min,
            rpm_run_max=cfg.rpm_run_max,
            traction_limit_n=float(profile.mu[i]) * mass * G * DRIVE_AXLE_LOAD_SHARE,
        )
        ok = tr.feasible & allowed[i][:, None, None] & allowed[i + 1][None, :, None]
        stage = np.where(
            ok,
            tr.fuel_l
            + cfg.lam_l_per_s * np.where(ok, tr.dt_s, 0.0)
            + cfg.brake_penalty_l_per_mj * np.where(ok, tr.service_brake_mj, 0.0),
            np.inf,
        )

        # Best way to arrive at (v0, gear g1): keep gear g1, or shift from the cheapest gear.
        j_min = J.min(axis=1)
        j_arg = J.argmin(axis=1)
        via_shift = (j_min + cfg.shift_penalty_l)[:, None]
        stay = J <= via_shift
        j_prev = np.where(stay, J, via_shift)
        prev_gear = np.where(stay, gidx[None, :], j_arg[:, None])

        total = j_prev[:, None, :] + stage
        best_v0 = total.argmin(axis=0)  # (nv1, ng)
        J = np.take_along_axis(total, best_v0[None, :, :], axis=0)[0]
        back_v[i] = best_v0
        back_g[i] = np.take_along_axis(prev_gear, best_v0, axis=0)

    final = J[0]
    if not np.isfinite(final.min()):
        raise PlanInfeasible(
            f"no feasible plan for route {profile.route_id!r} with payload {payload_kg:.0f} kg"
        )

    g_idx = int(final.argmin())
    v_idx = 0
    v_nodes = np.zeros(n + 1, dtype=int)
    g_used = np.zeros(n, dtype=int)
    for i in range(n - 1, -1, -1):
        g_used[i] = g_idx
        prev_v = int(back_v[i][v_idx, g_idx])
        prev_g = int(back_g[i][v_idx, g_idx])
        v_nodes[i] = prev_v
        v_idx, g_idx = prev_v, prev_g
    v_nodes[n] = 0

    # Re-evaluate the chosen path to report fuel and time per interval.
    v_plan = V[v_nodes]
    gear_plan = gears[g_used]
    dt = np.zeros(n)
    fuel = np.zeros(n)
    objective = 0.0
    for i in range(n):
        tr = vehicle.transition(
            v_plan[i],
            v_plan[i + 1],
            gear_plan[i],
            ds,
            profile.grade_pct[i],
            profile.crr[i],
            mass,
            a_acc_max=cfg.a_acc_max,
            a_dec_max=cfg.a_dec_max,
            rpm_run_min=cfg.rpm_run_min,
            rpm_run_max=cfg.rpm_run_max,
            traction_limit_n=float(profile.mu[i]) * mass * G * DRIVE_AXLE_LOAD_SHARE,
        )
        dt[i] = float(tr.dt_s)
        fuel[i] = float(tr.fuel_l)
        objective += (
            fuel[i]
            + cfg.lam_l_per_s * dt[i]
            + cfg.brake_penalty_l_per_mj * float(tr.service_brake_mj)
        )
    objective += cfg.shift_penalty_l * int((np.diff(gear_plan) != 0).sum())

    idle_l = float(
        vehicle.grams_to_litres(profile.dwell_s.sum() * vehicle.idle_fuel_gps)
    )
    return Plan(
        route_id=profile.route_id,
        payload_kg=payload_kg,
        config=cfg,
        s=profile.s.copy(),
        v=v_plan,
        gear=gear_plan,
        dt_s=dt,
        fuel_l=fuel,
        dwell_s=profile.dwell_s.copy(),
        idle_fuel_l=idle_l,
        objective=float(objective),
    )
