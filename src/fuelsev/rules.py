"""Rule-based detection: the transparent baseline.

Fixed thresholds on telemetry, each with a short duration requirement so that a single
noisy sample does not raise an alert. Thresholds are assumptions typical of fleet
telematics for heavy trucks; tune them per fleet in ``RuleConfig``.

The rules cover unsafe driving (overspeed, harsh braking, coasting in neutral, over-revving)
and machine faults (brake and coolant temperature, oil pressure, fuel loss, a fuel flow that
does not match the fuel map, overload, excessive idling).
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np

from .constants import DIESEL_DENSITY_KG_PER_L
from .engine import full_load_torque
from .fuelmap import FuelMap
from .schemas import Category, Severity, Telemetry


@dataclass(frozen=True)
class RuleConfig:
    overspeed_warn_ratio: float = 1.10
    overspeed_crit_ratio: float = 1.25
    overspeed_warn_s: float = 5.0
    overspeed_crit_s: float = 3.0
    harsh_brake_warn: float = -3.0  # m/s^2
    harsh_brake_crit: float = -4.5
    neutral_min_kmh: float = 20.0
    neutral_min_s: float = 4.0
    overrev_rpm: float = 2000.0
    overrev_s: float = 3.0
    brake_temp_warn_c: float = 300.0
    brake_temp_crit_c: float = 400.0
    coolant_warn_c: float = 105.0
    coolant_crit_c: float = 112.0
    coolant_s: float = 10.0
    oil_min_idle_kpa: float = 70.0  # minimum acceptable pressure at 600 rpm
    oil_min_slope_kpa_per_rpm: float = 0.08
    oil_s: float = 5.0
    fuel_loss_short_window_s: float = 60.0
    fuel_loss_short_l: float = 4.0
    fuel_loss_long_window_s: float = 600.0
    fuel_loss_long_l: float = 3.0
    fuel_flow_ratio: float = 1.15
    fuel_flow_window_s: float = 60.0
    overload_warn: float = 1.05  # fraction of rated payload
    overload_crit: float = 1.15
    idle_excess_s: float = 180.0


@dataclass(frozen=True)
class Detection:
    code: str
    severity: Severity
    category: Category
    message: str
    evidence: dict = field(default_factory=dict)


class _Timer:
    """Tracks how long a condition has been continuously true."""

    def __init__(self) -> None:
        self.since: float | None = None

    def update(self, active: bool, t: float) -> float:
        if not active:
            self.since = None
            return 0.0
        if self.since is None:
            self.since = t
        return t - self.since


class RuleEngine:
    def __init__(self, fuel_map: FuelMap, rated_payload_t: float, config: RuleConfig | None = None):
        self.cfg = config or RuleConfig()
        self.fuel_map = fuel_map
        self.rated_payload_t = rated_payload_t
        self._timers = {k: _Timer() for k in ("over_w", "over_c", "neutral", "rev", "cool_w", "cool_c", "oil", "idle")}
        self._hist: deque[Telemetry] = deque(maxlen=int(self.cfg.fuel_loss_long_window_s) + 5)
        self._flow: deque[tuple[float, float]] = deque(maxlen=int(self.cfg.fuel_flow_window_s) + 5)

    # ------------------------------------------------------------------
    def update(self, tel: Telemetry) -> list[Detection]:
        c = self.cfg
        t = tel.sim_t
        self._hist.append(tel)
        out: list[Detection] = []

        # --- unsafe driving -------------------------------------------------------------
        limit = tel.speed_limit_kmh
        if limit > 0:
            ratio = tel.speed_kmh / limit
            crit_for = self._timers["over_c"].update(ratio > c.overspeed_crit_ratio, t)
            warn_for = self._timers["over_w"].update(ratio > c.overspeed_warn_ratio, t)
            ev = {"speed_kmh": round(tel.speed_kmh, 1), "limit_kmh": round(limit, 1)}
            if crit_for >= c.overspeed_crit_s:
                out.append(Detection("OVERSPEED", "critical", "safety",
                                     f"Speed {tel.speed_kmh:.0f} km/h against a limit of {limit:.0f} km/h", ev))
            elif warn_for >= c.overspeed_warn_s:
                out.append(Detection("OVERSPEED", "warning", "safety",
                                     f"Speed {tel.speed_kmh:.0f} km/h against a limit of {limit:.0f} km/h", ev))

        if tel.accel_ms2 <= c.harsh_brake_warn:
            sev: Severity = "critical" if tel.accel_ms2 <= c.harsh_brake_crit else "warning"
            out.append(Detection("HARSH_BRAKE", sev, "safety",
                                 f"Harsh braking ({tel.accel_ms2:.1f} m/s2) at {tel.speed_kmh:.0f} km/h",
                                 {"accel_ms2": round(tel.accel_ms2, 2), "speed_kmh": round(tel.speed_kmh, 1)}))

        coasting = tel.gear == 0 and tel.speed_kmh > c.neutral_min_kmh
        if self._timers["neutral"].update(coasting, t) >= c.neutral_min_s:
            steep = tel.grade_pct < -2.0
            out.append(Detection("NEUTRAL_COASTING", "critical" if steep else "warning", "safety",
                                 "Coasting in neutral" + (" on a downhill" if steep else ""),
                                 {"speed_kmh": round(tel.speed_kmh, 1), "grade_pct": round(tel.grade_pct, 1)}))

        if self._timers["rev"].update(tel.rpm > c.overrev_rpm, t) >= c.overrev_s:
            out.append(Detection("OVER_REV", "warning", "efficiency", f"Engine over-revving at {tel.rpm:.0f} rpm",
                                 {"rpm": round(tel.rpm)}))

        # --- machine faults ---------------------------------------------------------------
        if tel.brake_temp_c >= c.brake_temp_warn_c:
            sev = "critical" if tel.brake_temp_c >= c.brake_temp_crit_c else "warning"
            out.append(Detection("BRAKE_OVERHEAT", sev, "fault", f"Brake temperature {tel.brake_temp_c:.0f} C",
                                 {"brake_temp_c": round(tel.brake_temp_c)}))

        cool_c = self._timers["cool_c"].update(tel.coolant_c > c.coolant_crit_c, t)
        cool_w = self._timers["cool_w"].update(tel.coolant_c > c.coolant_warn_c, t)
        if cool_c >= c.coolant_s or cool_w >= c.coolant_s:
            sev = "critical" if cool_c >= c.coolant_s else "warning"
            out.append(Detection("COOLANT_HIGH", sev, "fault", f"Coolant temperature {tel.coolant_c:.0f} C",
                                 {"coolant_c": round(tel.coolant_c, 1)}))

        if tel.engine_on:
            min_kpa = c.oil_min_idle_kpa + c.oil_min_slope_kpa_per_rpm * max(tel.rpm - 600.0, 0.0)
            if self._timers["oil"].update(tel.oil_kpa < min_kpa, t) >= c.oil_s:
                out.append(Detection("LOW_OIL_PRESSURE", "critical", "fault",
                                     f"Oil pressure {tel.oil_kpa:.0f} kPa, below {min_kpa:.0f} kPa at {tel.rpm:.0f} rpm",
                                     {"oil_kpa": round(tel.oil_kpa), "min_kpa": round(min_kpa)}))
        else:
            self._timers["oil"].update(False, t)

        out.extend(self._fuel_loss(tel))
        out.extend(self._fuel_flow(tel))

        # --- load and idling -----------------------------------------------------------------
        if self.rated_payload_t > 0 and tel.speed_kmh < 1.0 and tel.leg == "loaded":
            frac = tel.payload_t / self.rated_payload_t
            if frac >= c.overload_warn:
                sev = "critical" if frac >= c.overload_crit else "warning"
                out.append(Detection("OVERLOAD", sev, "load",
                                     f"Payload {tel.payload_t:.1f} t is {100 * (frac - 1):.0f}% over the rated "
                                     f"{self.rated_payload_t:.1f} t",
                                     {"payload_t": round(tel.payload_t, 2), "over_pct": round(100 * (frac - 1), 1)}))

        idle = tel.engine_on and tel.speed_kmh < 0.5
        if self._timers["idle"].update(idle, t) >= c.idle_excess_s:
            out.append(Detection("EXCESS_IDLE", "info", "efficiency", "Engine idling for over 3 minutes",
                                 {"idle_s": round(self._timers["idle"].update(idle, t))}))
        return out

    # ------------------------------------------------------------------
    def _median_at(self, samples: list[Telemetry], attr: str) -> float:
        return float(np.median([getattr(s, attr) for s in samples]))

    def _fuel_loss(self, tel: Telemetry) -> list[Detection]:
        c = self.cfg
        out: list[Detection] = []
        for window_s, thr_l, label in (
            (c.fuel_loss_short_window_s, c.fuel_loss_short_l, "short"),
            (c.fuel_loss_long_window_s, c.fuel_loss_long_l, "long"),
        ):
            n = int(window_s)
            if len(self._hist) < n:
                continue
            hist = list(self._hist)
            old, new = hist[-n:][:5], hist[-5:]
            level_drop = self._median_at(old, "fuel_level_l") - self._median_at(new, "fuel_level_l")
            burned = new[-1].fuel_total_l - old[0].fuel_total_l
            lost = level_drop - burned
            if lost > thr_l:
                out.append(Detection("FUEL_LOSS", "critical", "fault",
                                     f"Tank level fell {lost:.1f} L more than the engine burned in "
                                     f"{window_s / 60:.0f} min: possible leak or theft",
                                     {"unexplained_l": round(lost, 1), "window": label}))
                break
        return out

    def _fuel_flow(self, tel: Telemetry) -> list[Detection]:
        """Compare measured fuel flow with what the fuel map predicts at the same rpm and torque."""
        c = self.cfg
        if tel.engine_on and tel.gear > 0 and tel.torque_pct > 15.0 and abs(tel.accel_ms2) < 1.5 and tel.rpm > 700:
            torque = tel.torque_pct / 100.0 * float(full_load_torque(tel.rpm))
            expected_lph = float(self.fuel_map.rate_gps(tel.rpm, torque)) * 3600.0 / (DIESEL_DENSITY_KG_PER_L * 1000.0)
            if expected_lph > 5.0:
                self._flow.append((tel.fuel_lph, expected_lph))
        if len(self._flow) >= 30:
            meas = sum(m for m, _ in self._flow)
            exp = sum(e for _, e in self._flow)
            ratio = meas / exp
            if ratio > c.fuel_flow_ratio:
                self._flow.clear()
                return [Detection("FUEL_FLOW_HIGH", "warning", "fault",
                                  f"Fuel flow is {100 * (ratio - 1):.0f}% above the engine map: check injectors and air intake",
                                  {"ratio": round(ratio, 3)})]
        return []
