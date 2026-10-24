"""In-cab coaching: short, prioritised, rate-limited messages for the operator.

Two kinds of advice:

* Safety and fault warnings, taken from the rule detections (overspeed, coasting in neutral,
  hot brakes, low oil, overload...).
* Fuel advice, computed from the fuel map and the DP plan: "shift to gear 7", "ease off to
  45 km/h", "lift off, stop ahead", "use gear 5 on this descent".

Only one message is shown at a time. Safety beats faults beat efficiency, and each message
has a cooldown, so the operator is not nagged.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .i18n import render
from .planner import Plan
from .routes import KIND_BUMP, KIND_HAIRPIN, KIND_STOP, RouteProfile, Scenario
from .rules import Detection
from .schemas import CoachMessage, Telemetry
from .vehicle import Vehicle

DEFAULT_CRR = 0.011


@dataclass
class RouteContext:
    scenario: Scenario
    leg: str
    profile: RouteProfile
    plan: Plan | None
    payload_kg: float


@dataclass(frozen=True)
class _Candidate:
    code: str
    value: float  # ranking within a priority: bigger first
    params: dict


_DEFAULT_COOLDOWN_S = 30.0
_COOLDOWNS = {
    "OVERSPEED": 20.0,
    "SMOOTH_BRAKE": 20.0,
    "NO_NEUTRAL": 25.0,
    "BRAKE_HOT": 60.0,
    "ENGINE_HOT": 60.0,
    "LOW_OIL": 60.0,
    "OVERLOAD": 300.0,
    "FUEL_LOSS": 300.0,
    "EASE_OFF": 60.0,
    "IDLE_OFF": 120.0,
    "DESCENT_GEAR": 60.0,
}


class Coach:
    def __init__(self, machine_id: str, vehicle: Vehicle, lang: str = "en", min_gap_s: float = 6.0):
        self.machine_id = machine_id
        self.veh = vehicle
        self.lang = lang
        self.min_gap_s = min_gap_s
        self._last_any = -1e9
        self._last_code: dict[str, float] = {}
        self._gear_votes = 0
        self._descent_votes = 0
        self._warned_nodes: set[tuple[str, int]] = set()
        self._idle_s = 0.0
        # exposed so the agent can track how far gear choice is from optimal
        self.last_fuel_pair: tuple[float, float] | None = None

    def reset_leg(self) -> None:
        """Forget per-leg state (which bumps were already announced) when a new leg starts."""
        self._warned_nodes.clear()
        self._gear_votes = 0
        self._descent_votes = 0

    # ------------------------------------------------------------------
    def advise(
        self, tel: Telemetry, ctx: RouteContext | None, detections: list[Detection]
    ) -> CoachMessage | None:
        cands: list[_Candidate] = []
        cands += self._from_detections(tel, detections)
        cands += self._lookahead(tel, ctx)
        cands += self._gear(tel, ctx)
        cands += self._speed_vs_plan(tel, ctx)
        cands += self._descent(tel)
        cands += self._idle(tel)
        if not cands:
            return None

        def prio(c: _Candidate) -> int:
            return render(c.code, self.lang, **_dummy(c))[0]

        cands.sort(key=lambda c: (prio(c), -c.value))
        for c in cands:
            p = prio(c)
            cooldown = _COOLDOWNS.get(c.code, _DEFAULT_COOLDOWN_S)
            if tel.sim_t - self._last_code.get(c.code, -1e9) < cooldown:
                continue
            gap = 2.0 if p == 1 else self.min_gap_s
            if tel.sim_t - self._last_any < gap:
                continue
            _, text = render(c.code, self.lang, **c.params)
            self._last_any = tel.sim_t
            self._last_code[c.code] = tel.sim_t
            return CoachMessage(
                machine_id=self.machine_id,
                ts=tel.ts,
                sim_t=tel.sim_t,
                code=c.code,
                priority=p,
                lang=self.lang,
                text=text,
                params={k: v for k, v in c.params.items() if isinstance(v, (int, float, str))},
            )
        return None

    # ------------------------------------------------------------------ safety and faults
    def _from_detections(self, tel: Telemetry, detections: list[Detection]) -> list[_Candidate]:
        out: list[_Candidate] = []
        for d in detections:
            if d.code == "OVERSPEED":
                out.append(_Candidate("OVERSPEED", 10, {"limit": round(tel.speed_limit_kmh)}))
            elif d.code == "NEUTRAL_COASTING":
                out.append(_Candidate("NO_NEUTRAL", 10, {}))
            elif d.code == "HARSH_BRAKE":
                out.append(_Candidate("SMOOTH_BRAKE", 5, {}))
            elif d.code == "BRAKE_OVERHEAT":
                out.append(_Candidate("BRAKE_HOT", 10, {"temp": round(tel.brake_temp_c)}))
            elif d.code == "COOLANT_HIGH":
                out.append(_Candidate("ENGINE_HOT", 10, {"temp": round(tel.coolant_c)}))
            elif d.code == "LOW_OIL_PRESSURE":
                out.append(_Candidate("LOW_OIL", 20, {}))
            elif d.code == "OVERLOAD":
                out.append(_Candidate("OVERLOAD", 5, {"pct": round(float(d.evidence.get("over_pct", 0)))}))
            elif d.code == "FUEL_LOSS":
                out.append(_Candidate("FUEL_LOSS", 5, {}))
        return out

    # ------------------------------------------------------------------ route look-ahead
    def _lookahead(self, tel: Telemetry, ctx: RouteContext | None) -> list[_Candidate]:
        if ctx is None or tel.speed_kmh < 8.0:
            return []
        prof = ctx.profile
        i = prof.interval_index(tel.pos_m)
        out: list[_Candidate] = []
        hi = min(i + 9, prof.n_nodes)  # about 200 m ahead
        for j in range(i + 1, hi):
            d = float(prof.s[j] - tel.pos_m)
            kind = int(prof.kind[j])
            if d <= 20.0 or (("n", j) in self._warned_nodes):
                continue
            if kind in (KIND_BUMP, KIND_HAIRPIN):
                limit = prof.cap_ms[j] * 3.6
                if d <= 150.0 and tel.speed_kmh > limit + 5.0:
                    self._warned_nodes.add(("n", j))
                    out.append(_Candidate("BUMP_AHEAD", 8, {"dist": int(round(d / 10) * 10), "limit": int(round(limit))}))
            elif kind == KIND_STOP and j < prof.n_nodes - 1:
                if d <= 200.0 and tel.speed_kmh > 25.0 and tel.throttle_pct > 8.0:
                    self._warned_nodes.add(("n", j))
                    out.append(_Candidate("STOP_AHEAD", 6, {"dist": int(round(d / 10) * 10)}))
        return out

    # ------------------------------------------------------------------ gear advice
    def _gear(self, tel: Telemetry, ctx: RouteContext | None) -> list[_Candidate]:
        self.last_fuel_pair = None
        if not tel.engine_on or tel.gear <= 0 or tel.speed_kmh < 5.0:
            self._gear_votes = 0
            return []
        crr = self._crr(tel, ctx)
        mass = self.veh.mass(tel.payload_t * 1000.0)
        gears, fuel, rpm, feas, demand = self.veh.gear_options(
            tel.speed_kmh / 3.6, tel.accel_ms2, tel.grade_pct, crr, mass
        )
        g = min(max(tel.gear, 1), self.veh.n_gears)
        if demand[g - 1] <= 0:  # overrun or braking: nothing to save
            self._gear_votes = 0
            return []

        # straining: close to full torque, and a lower gear has torque in hand
        if tel.torque_pct > 97.0 and g > 1:
            lower = [k for k in range(g - 1) if feas[k]]
            if lower:
                self._gear_votes = 0
                return [_Candidate("SHIFT_DOWN", 100, {"gear": int(gears[max(lower)])})]

        if not feas.any():
            self._gear_votes = 0
            return []
        scored = np.where(feas, fuel, np.inf)
        best = int(np.argmin(scored))
        cur = float(fuel[g - 1])
        self.last_fuel_pair = (cur, float(scored[best]))
        saving = 1.0 - float(scored[best]) / cur if cur > 0 else 0.0
        if best + 1 != g and saving >= 0.08:
            self._gear_votes += 1
        else:
            self._gear_votes = 0
        if self._gear_votes >= 3:
            code = "SHIFT_UP" if best + 1 > g else "SHIFT_DOWN"
            return [_Candidate(code, 100 * saving, {"gear": int(gears[best]), "pct": int(round(100 * saving))})]
        return []

    # ------------------------------------------------------------------ speed advice from the plan
    def _speed_vs_plan(self, tel: Telemetry, ctx: RouteContext | None) -> list[_Candidate]:
        if ctx is None or ctx.plan is None or tel.speed_kmh < 15.0 or tel.accel_ms2 < -0.3:
            return []
        v_plan = ctx.plan.target_speed(tel.pos_m)
        v = tel.speed_kmh / 3.6
        if v_plan < 3.0 or v < 1.15 * v_plan + 1.0:
            return []
        crr = self._crr(tel, ctx)
        mass = self.veh.mass(tel.payload_t * 1000.0)
        now = self._steady_fuel_per_m(v, tel.grade_pct, crr, mass)
        plan = self._steady_fuel_per_m(v_plan, tel.grade_pct, crr, mass)
        if now is None or plan is None or now <= 0:
            return []
        saving = 1.0 - plan / now
        if saving >= 0.08:
            return [_Candidate("EASE_OFF", 100 * saving, {"target": int(round(v_plan * 3.6)), "pct": int(round(100 * saving))})]
        return []

    def _steady_fuel_per_m(self, v: float, grade_pct: float, crr: float, mass: float) -> float | None:
        gears, fuel, _, feas, demand = self.veh.gear_options(v, 0.0, grade_pct, crr, mass)
        ok = feas & (demand > 0)
        if not ok.any():
            return None
        return float(np.min(fuel[ok])) / v

    # ------------------------------------------------------------------ descents and idling
    def _descent(self, tel: Telemetry) -> list[_Candidate]:
        if tel.grade_pct < -3.0 and tel.gear > 0 and tel.brake_pct > 25.0 and tel.speed_kmh > 15.0:
            self._descent_votes += 1
        else:
            self._descent_votes = 0
        if self._descent_votes >= 5:
            gs = np.arange(1, self.veh.n_gears + 1)
            ok = self.veh.rpm_at(tel.speed_kmh / 3.6, gs) <= 1900.0
            if ok.any():
                g = int(gs[ok].min())
                if g < tel.gear:
                    return [_Candidate("DESCENT_GEAR", 50, {"gear": g})]
        return []

    def _idle(self, tel: Telemetry) -> list[_Candidate]:
        if tel.engine_on and tel.speed_kmh < 0.5:
            self._idle_s += 1.0
        else:
            self._idle_s = 0.0
        if self._idle_s >= 60.0:
            return [_Candidate("IDLE_OFF", 30, {})]
        return []

    # ------------------------------------------------------------------
    def _crr(self, tel: Telemetry, ctx: RouteContext | None) -> float:
        if ctx is None:
            return DEFAULT_CRR
        return float(ctx.profile.crr[ctx.profile.interval_index(tel.pos_m)])


def _dummy(c: _Candidate) -> dict:
    """Parameters used only to look up a message's priority."""
    base = {"gear": 0, "pct": 0, "target": 0, "limit": 0, "dist": 0, "temp": 0}
    base.update(c.params)
    return base
