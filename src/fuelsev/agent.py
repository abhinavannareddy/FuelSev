"""The in-cab agent.

Consumes the machine's 1 Hz telemetry and produces:

* cab messages for the operator (coaching and warnings),
* alerts for the supervisor (rules, plus the Isolation Forest if a model is loaded),
* a trip summary at the end of every haul cycle (fuel, CO2e inputs, eco score).

It runs on the machine, so alerts are raised even when the cellular link is down; delivery
to the supervisor is the job of the transport layer (``machine.py``), which buffers them.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

from .anomaly import AnomalyModel
from .coach import Coach, RouteContext
from .features import HOP_S, WINDOW_S, is_driving, window_features
from .metrics import eco_score
from .planner import PlannerConfig, plan_leg
from .routes import SCENARIOS
from .rules import Detection, RuleConfig, RuleEngine
from .schemas import Alert, CoachMessage, Severity, Telemetry, TripSummary
from .vehicle import Vehicle

IDLE_ALLOWANCE_S = 60.0  # waiting this long at a stop (toll, signal, crossing) is not held against the driver

_SEVERITY_RANK = {"info": 0, "warning": 1, "critical": 2}

_DEFAULT_ALERT_COOLDOWN_S = 60.0
_ALERT_COOLDOWNS = {
    "HARSH_BRAKE": 15.0,
    "OVERSPEED": 60.0,
    "NEUTRAL_COASTING": 60.0,
    "OVER_REV": 120.0,
    "BRAKE_OVERHEAT": 120.0,
    "COOLANT_HIGH": 120.0,
    "LOW_OIL_PRESSURE": 120.0,
    "FUEL_LOSS": 600.0,
    "FUEL_FLOW_HIGH": 600.0,
    "OVERLOAD": 600.0,
    "EXCESS_IDLE": 600.0,
    "ANOMALOUS_DRIVING": 120.0,
}


@dataclass
class AgentOutput:
    coach: CoachMessage | None = None
    alerts: list[Alert] = field(default_factory=list)
    trip: TripSummary | None = None


class _TripAcc:
    def __init__(self, tel: Telemetry, base: Telemetry) -> None:
        self.machine_id = tel.machine_id
        self.trip_no = tel.trip_no
        self.route_id = tel.route_id
        self.start_ts = base.ts
        self.start_t = base.sim_t
        self.fuel0 = base.fuel_total_l
        self.odo0 = base.odometer_km
        self.idle0 = base.idle_s
        self.idle_fuel0 = base.idle_fuel_l
        self.engine0 = base.engine_s
        self.payload_max = 0.0
        self.loaded_odo_start: float | None = None
        self.loaded_odo_end: float | None = None
        self.driving_s = 0.0
        self.overspeed_s = 0.0
        self.harsh = 0
        self._last_harsh_t = -1e9
        self.fuel_cur = 0.0
        self.fuel_best = 0.0
        self.idle_spell_s = 0.0
        self.idle_excess_s = 0.0  # idling beyond the first minute of each stop
        self.last = tel

    def observe(self, tel: Telemetry, detections: list[Detection], pair: tuple[float, float] | None) -> None:
        self.last = tel
        self.payload_max = max(self.payload_max, tel.payload_t)
        if tel.leg == "loaded":
            if self.loaded_odo_start is None:
                self.loaded_odo_start = tel.odometer_km
            self.loaded_odo_end = tel.odometer_km
        if tel.speed_kmh > 1.0:
            self.driving_s += 1.0
            if tel.speed_limit_kmh > 0 and tel.speed_kmh > 1.10 * tel.speed_limit_kmh:
                self.overspeed_s += 1.0
        if tel.engine_on and tel.speed_kmh < 0.5:
            self.idle_spell_s += 1.0
            if self.idle_spell_s > IDLE_ALLOWANCE_S:
                self.idle_excess_s += 1.0
        else:
            self.idle_spell_s = 0.0
        if tel.accel_ms2 <= -3.0 and tel.sim_t - self._last_harsh_t >= 5.0:
            self.harsh += 1
            self._last_harsh_t = tel.sim_t
        if pair is not None:
            self.fuel_cur += pair[0]
            self.fuel_best += pair[1]

    def summary(self) -> TripSummary:
        t = self.last
        driving = max(self.driving_s, 1.0)
        engine_s = max(t.engine_s - self.engine0, 1.0)
        idle_s = t.idle_s - self.idle0
        gear_excess = self.fuel_cur / self.fuel_best - 1.0 if self.fuel_best > 0 else 0.0
        score = eco_score(
            gear_excess=gear_excess,
            harsh_per_10min=self.harsh / (driving / 600.0),
            overspeed_frac=self.overspeed_s / driving,
            idle_frac=self.idle_excess_s / engine_s,
        )
        loaded_km = 0.0
        if self.loaded_odo_start is not None and self.loaded_odo_end is not None:
            loaded_km = self.loaded_odo_end - self.loaded_odo_start
        return TripSummary(
            machine_id=self.machine_id,
            trip_no=self.trip_no,
            route_id=self.route_id,
            start_ts=self.start_ts,
            end_ts=t.ts,
            duration_s=t.sim_t - self.start_t,
            payload_t=self.payload_max,
            distance_km=t.odometer_km - self.odo0,
            loaded_distance_km=loaded_km,
            fuel_l=t.fuel_total_l - self.fuel0,
            idle_s=idle_s,
            idle_fuel_l=t.idle_fuel_l - self.idle_fuel0,
            harsh_brake_events=self.harsh,
            overspeed_s=self.overspeed_s,
            eco_score=score,
        )


class CabAgent:
    def __init__(
        self,
        machine_id: str,
        *,
        vehicle: Vehicle | None = None,
        anomaly_model: AnomalyModel | None = None,
        lang: str = "en",
        rule_config: RuleConfig | None = None,
        use_plan: bool = True,
        planner_config: PlannerConfig | None = None,
    ):
        self.machine_id = machine_id
        self.veh = vehicle or Vehicle()
        self.model = anomaly_model
        self.use_plan = use_plan
        self.planner_config = planner_config or PlannerConfig()
        self.rules = RuleEngine(self.veh.fuel_map, self.veh.params.payload_rated_kg / 1000.0, rule_config)
        self.coach = Coach(machine_id, self.veh, lang)
        self._window: deque[Telemetry] = deque(maxlen=WINDOW_S)
        self._since_hop = 0
        self._anomalous_run = 0
        self._last_alert: dict[str, tuple[float, str]] = {}
        self._trip: _TripAcc | None = None
        self._ctx: RouteContext | None = None
        self._ctx_key: tuple | None = None
        self._plans: dict[tuple, object] = {}
        self._profiles: dict[tuple, object] = {}
        self._planned = False
        self._prev: Telemetry | None = None

    # ------------------------------------------------------------------
    def update(self, tel: Telemetry) -> AgentOutput:
        out = AgentOutput()

        if self._trip is None or tel.trip_no != self._trip.trip_no:
            if self._trip is not None:
                out.trip = self._trip.summary()
            base = self._prev or tel
            self._trip = _TripAcc(tel, base)

        self._refresh_context(tel)
        detections = self.rules.update(tel)
        for d in detections:
            alert = self._maybe_alert(tel, d)
            if alert:
                out.alerts.append(alert)

        extra = self._anomaly(tel)
        if extra:
            alert = self._maybe_alert(tel, extra)
            if alert:
                out.alerts.append(alert)

        out.coach = self.coach.advise(tel, self._ctx, detections)
        self._trip.observe(tel, detections, self.coach.last_fuel_pair)
        self._prev = tel
        return out

    def finish(self) -> TripSummary | None:
        """Call at the end of the stream to close the last trip."""
        if self._trip is None:
            return None
        summary = self._trip.summary()
        self._trip = None
        return summary

    # ------------------------------------------------------------------ internals
    def _refresh_context(self, tel: Telemetry) -> None:
        """Keep the route context current: profile per leg, plan once the truck is moving."""
        scenario = SCENARIOS.get(tel.route_id)
        if scenario is None:
            self._ctx = None
            self._ctx_key = None
            return
        leg_key = (tel.route_id, tel.leg)
        if self._ctx is None or self._ctx_key != leg_key:
            self._ctx = RouteContext(scenario, tel.leg, self._profile(tel), None, 0.0)
            self._ctx_key = leg_key
            self._planned = False
            self.coach.reset_leg()
        if self.use_plan and not self._planned and tel.speed_kmh >= 3.0:
            bucket = round(tel.payload_t * 2) / 2.0 if tel.leg == "loaded" else 0.0
            pkey = (tel.route_id, tel.leg, bucket)
            plan = self._plans.get(pkey)
            if plan is None:
                route = scenario.leg(tel.leg)
                plan = plan_leg(self.veh, route.profile(), bucket * 1000.0, self.planner_config)
                self._plans[pkey] = plan
            self._ctx.plan = plan
            self._ctx.payload_kg = bucket * 1000.0
            self._planned = True

    def _profile(self, tel: Telemetry):
        key = (tel.route_id, tel.leg)
        prof = self._profiles.get(key)
        if prof is None:
            prof = SCENARIOS[tel.route_id].leg(tel.leg).profile()
            self._profiles[key] = prof
        return prof

    def _maybe_alert(self, tel: Telemetry, d: Detection) -> Alert | None:
        cooldown = _ALERT_COOLDOWNS.get(d.code, _DEFAULT_ALERT_COOLDOWN_S)
        last = self._last_alert.get(d.code)
        if last is not None:
            last_t, last_sev = last
            escalated = _SEVERITY_RANK[d.severity] > _SEVERITY_RANK[last_sev]
            if tel.sim_t - last_t < cooldown and not escalated:
                return None
        self._last_alert[d.code] = (tel.sim_t, d.severity)
        return Alert(
            id=Alert.make_id(tel.machine_id, d.code, tel.sim_t),
            machine_id=tel.machine_id,
            ts=tel.ts,
            sim_t=tel.sim_t,
            code=d.code,
            severity=d.severity,
            category=d.category,
            message=d.message,
            route_id=tel.route_id,
            pos_m=tel.pos_m,
            evidence=d.evidence,
        )

    def _anomaly(self, tel: Telemetry) -> Detection | None:
        """Score a 10 s window every 5 s. Two anomalous windows in a row raise an alert."""
        if self.model is None:
            return None
        self._window.append(tel)
        self._since_hop += 1
        if len(self._window) < WINDOW_S or self._since_hop < HOP_S:
            return None
        self._since_hop = 0
        window = list(self._window)
        if not is_driving(window):
            self._anomalous_run = 0
            return None
        x = window_features(window)
        score = float(self.model.scores(x)[0])
        if score < self.model.threshold:
            self._anomalous_run += 1
        else:
            self._anomalous_run = 0
        if self._anomalous_run >= 2:
            top = self.model.explain(x, 3)
            names = ", ".join(f"{n} ({z:+.1f})" for n, z in top)
            sev: Severity = "warning"
            return Detection(
                "ANOMALOUS_DRIVING",
                sev,
                "safety",
                f"Driving pattern is unusual compared with normal operation: {names}",
                {"score": round(score, 3), "threshold": round(self.model.threshold, 3), "top_features": names},
            )
        return None
