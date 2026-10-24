
import string

import numpy as np
import pytest

from fuelsev.constants import CO2_KG_PER_L_DIESEL
from fuelsev.fuelmap import FuelMap, build_synthetic_fuel_map, load_default_fuel_map
from fuelsev.i18n import LANGS, MESSAGES, render
from fuelsev.metrics import aggregate_trips, co2e_kg, eco_score
from fuelsev.mqtt import ReliableSender, parse_topic, topic
from fuelsev.planner import plan_leg
from fuelsev.routes import SCENARIOS
from fuelsev.schemas import Alert, Telemetry, TripSummary
from fuelsev.vehicle import Vehicle


def test_co2_factor_is_documented_value():
    assert CO2_KG_PER_L_DIESEL == pytest.approx(2.68, abs=0.01)
    assert co2e_kg(100.0) == pytest.approx(268.0, abs=1.0)


def test_fuel_map_roundtrip_and_interpolation(tmp_path):
    fm = build_synthetic_fuel_map()
    p = tmp_path / "m.csv"
    fm.to_csv(p)
    fm2 = FuelMap.from_csv(p)
    assert float(fm2.rate_gps(1400, 800)) == pytest.approx(float(fm.rate_gps(1400, 800)), rel=1e-6)
    # fuel rate rises with torque at fixed rpm, and is positive
    lo, hi = float(fm.rate_gps(1400, 300)), float(fm.rate_gps(1400, 1200))
    assert 0 < lo < hi
    # off-grid points are clamped rather than raising
    assert np.isfinite(float(fm.rate_gps(5000, 5000)))
    assert load_default_fuel_map().rate_gps(1200, 500) > 0


def test_bsfc_is_plausible_for_a_diesel():
    fm = load_default_fuel_map()
    b = float(fm.bsfc_gkwh(1300, 1100))
    assert 190 < b < 260


def test_vehicle_climbing_needs_more_force_than_flat():
    v = Vehicle()
    m = v.mass(20000.0)
    flat = float(v.resistance_n(10.0, m, 0.0, 0.011))
    hill = float(v.resistance_n(10.0, m, 8.0, 0.011))
    assert hill > 3 * flat


def test_planner_is_feasible_and_cheaper_than_flooring_it():
    v = Vehicle()
    prof = SCENARIOS["nh_corridor"].loaded.profile()
    plan = plan_leg(v, prof, 22000.0)
    assert np.all(plan.v <= prof.cap_ms + 1e-6)
    assert plan.v[0] == 0.0 and plan.v[-1] == 0.0
    assert plan.moving_fuel_l > 0
    # a plan that values time more must not use less fuel than one that values it less
    from fuelsev.planner import PlannerConfig
    hurried = plan_leg(v, prof, 22000.0, PlannerConfig(lam_l_per_s=0.05))
    relaxed = plan_leg(v, prof, 22000.0, PlannerConfig(lam_l_per_s=0.0005))
    assert hurried.moving_fuel_l >= relaxed.moving_fuel_l - 1e-6
    assert hurried.moving_time_s <= relaxed.moving_time_s + 1e-6


def test_sim_counters_are_monotonic(short_run):
    tel = short_run.telemetry
    for field in ("fuel_total_l", "odometer_km", "engine_s", "idle_s"):
        vals = np.array([getattr(t, field) for t in tel])
        assert np.all(np.diff(vals) >= -1e-9), field
    assert max(t.speed_kmh for t in tel) < 90


def test_sim_is_deterministic():
    from fuelsev.simulator import Simulation
    a = Simulation("nh_corridor", "typical", seed=5).run().telemetry[600]
    b = Simulation("nh_corridor", "typical", seed=5).run().telemetry[600]
    assert a.fuel_total_l == b.fuel_total_l and a.speed_kmh == b.speed_kmh


def test_driving_styles_are_ordered_by_fuel():
    from fuelsev.simulator import Simulation
    fuel = {}
    for d in ("planner", "typical", "aggressive"):
        r = Simulation("nh_corridor", d, seed=2).run().telemetry
        fuel[d] = r[-1].fuel_total_l
    assert fuel["planner"] < fuel["typical"] < fuel["aggressive"]


def test_plan_replay_matches_simulation_within_five_percent():
    from fuelsev.simulator import Simulation
    v = Vehicle()
    sim = Simulation("ghat_climb", "planner", seed=1, vehicle=v, hazards=False)
    res = sim.run().telemetry
    loaded_fuel = [t for t in res if t.leg == "loaded"][-1].fuel_total_l
    plan = plan_leg(v, SCENARIOS["ghat_climb"].loaded.profile(), res[len(res) // 2].payload_t * 1000.0)
    assert abs(loaded_fuel - plan.total_fuel_l) / plan.total_fuel_l < 0.05


def test_rules_flag_neutral_coasting(short_run):
    from fuelsev.rules import RuleEngine
    base = next(t for t in short_run.telemetry if t.speed_kmh > 30)
    veh = Vehicle()
    eng = RuleEngine(veh.fuel_map, 25.0)
    codes = set()
    for i in range(30):
        t = base.model_copy(update={"gear": 0, "sim_t": base.sim_t + i, "grade_pct": -5.0, "speed_kmh": 40.0,
                                    "throttle_pct": 0.0, "fuel_lph": 0.0})
        codes |= {d.code for d in eng.update(t)}
    assert "NEUTRAL_COASTING" in codes


def test_rules_are_quiet_for_the_planner_driver():
    from fuelsev.agent import CabAgent
    from fuelsev.simulator import Simulation
    agent = CabAgent("T-2", use_plan=False)
    sim = Simulation("nh_corridor", "planner", seed=4, trips=1, machine_id="T-2")
    safety = [a for tel, _ in sim.stream() for a in agent.update(tel).alerts if a.category == "safety"]
    assert len(safety) <= 3


def test_anomaly_model_train_save_load(tmp_path):
    from fuelsev.anomaly import AnomalyModel, train_anomaly_model
    from fuelsev.features import N_FEATURES
    rng = np.random.default_rng(0)
    X = rng.normal(size=(600, N_FEATURES))
    m = train_anomaly_model(X, false_alarm_rate=0.05)
    flagged = m.is_anomalous(X).mean()
    assert 0.03 < flagged < 0.08
    outlier = np.full((1, N_FEATURES), 8.0)
    assert m.is_anomalous(outlier)[0]
    p = tmp_path / "m.joblib"
    m.save(p)
    m2 = AnomalyModel.load(p)
    assert np.allclose(m.scores(X[:20]), m2.scores(X[:20]))
    assert len(m.explain(outlier[0], 3)) == 3
    with pytest.raises(ValueError):
        train_anomaly_model(X[:10])


def test_i18n_every_message_has_all_languages_and_matching_placeholders():
    fmt = string.Formatter()
    for code, (prio, texts) in MESSAGES.items():
        assert prio in (1, 2, 3)
        assert set(texts) >= set(LANGS), code
        ph = {lang: {f for _, f, _, _ in fmt.parse(texts[lang]) if f} for lang in LANGS}
        assert ph["hi"] == ph["en"] == ph["te"], code


def test_i18n_render_fills_parameters():
    _, text = render("OVERSPEED", "en", limit=40)
    assert "40" in text
    _, hi = render("OVERSPEED", "hi", limit=40)
    assert "40" in hi


def test_eco_score_bounds_and_direction():
    assert eco_score(0, 0, 0, 0) == 100.0
    assert eco_score(1, 100, 1, 1) == 0.0
    assert eco_score(0.05, 1, 0.02, 0.0) > eco_score(0.10, 3, 0.10, 0.05)


def test_fleet_aggregate_is_ratio_of_sums():
    def trip(n, fuel, payload):
        return TripSummary(machine_id="M", trip_no=n, route_id="r", start_ts=0, end_ts=1, duration_s=1, payload_t=payload,
                           distance_km=10, loaded_distance_km=5, fuel_l=fuel, idle_s=0, idle_fuel_l=0,
                           harsh_brake_events=0, overspeed_s=0, eco_score=80)
    agg = aggregate_trips([trip(1, 10, 10), trip(2, 30, 30)])
    assert agg["l_per_tonne"] == pytest.approx(1.0)
    agg2 = aggregate_trips([trip(1, 10, 20), trip(2, 30, 10)])
    assert agg2["l_per_tonne"] == pytest.approx(40 / 30, abs=1e-3)  # not the mean of 0.5 and 3.0
    assert agg2["co2e_kg"] == pytest.approx(40 * CO2_KG_PER_L_DIESEL, abs=0.01)
    assert aggregate_trips([])["l_per_tonne"] is None


def test_topics_and_validation():
    assert topic("HT-01", "alerts") == "fuelsev/v1/HT-01/alerts"
    assert parse_topic("fuelsev/v1/HT-01/alerts") == ("HT-01", "alerts")
    for bad in ("fuelsev/v1/HT-01/other", "x/v1/HT-01/alerts", "fuelsev/v1/a/b/alerts", "fuelsev/v1/../alerts", "fuelsev/v1/HT 01/alerts"):
        assert parse_topic(bad) is None
    with pytest.raises(ValueError):
        topic("bad id!", "alerts")


def test_alert_ids_are_deterministic():
    assert Alert.make_id("M", "OVERSPEED", 12.0) == Alert.make_id("M", "OVERSPEED", 12.0)
    assert Alert.make_id("M", "OVERSPEED", 12.0) != Alert.make_id("N", "OVERSPEED", 12.0)


def test_schema_rejects_bad_machine_id():
    with pytest.raises(ValueError):
        Telemetry.model_validate({"machine_id": "../etc", "ts": 0})


def test_reliable_sender_preserves_order_and_flushes():
    up = {"on": False}
    sent = []

    def pub(t, p, q, r):
        if not up["on"]:
            return False
        sent.append(p)
        return True

    s = ReliableSender(pub)
    s.send("t", "1")
    s.send("t", "2")
    assert len(s) == 2 and sent == []
    up["on"] = True
    s.send("t", "3")  # must not overtake the queued messages
    assert sent == []
    assert s.flush() == 3
    assert sent == ["1", "2", "3"]
    s.send("t", "4")
    assert sent[-1] == "4" and len(s) == 0


def test_reliable_sender_drops_oldest_when_full():
    s = ReliableSender(lambda *a: False, maxlen=3)
    for i in range(5):
        s.send("t", str(i))
    assert len(s) == 3 and s.dropped == 2
    assert s._queue[0][1] == "2"
