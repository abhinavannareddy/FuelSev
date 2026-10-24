import time

from fastapi.testclient import TestClient

from fuelsev.agent import CabAgent
from fuelsev.api import Settings, create_app
from fuelsev.demo import _Uplink
from fuelsev.simulator import Simulation


def _app(tmp_path, **kw):
    return create_app(Settings(db_path=str(tmp_path / "w.db"), ingest=False, **kw))


def test_dashboard_and_cab_pages_are_served_without_a_key(tmp_path):
    c = TestClient(_app(tmp_path, api_key="s3cret"))
    for path, marker in (("/", "Fleet control"), ("/cab", "Cab display")):
        r = c.get(path)
        assert r.status_code == 200 and marker in r.text
        assert "default-src 'self'" in r.headers["content-security-policy"]
    assert c.get("/machines").status_code == 401
    assert c.get("/health").json()["auth_required"] is True


def test_uplink_queues_alerts_in_a_dead_zone_and_delivers_them_in_order(tmp_path):
    app = _app(tmp_path)
    ing, store = app.state.ingestor, app.state.store
    up = _Uplink(ing, "HT-A")
    sim = Simulation("urban_mixed", "aggressive", machine_id="HT-A", seed=1, trips=1, t0_wall=time.time())
    agent = CabAgent("HT-A", use_plan=False)
    up.set_online(False)
    sent = 0
    for tel, _ in sim.stream():
        out = agent.update(tel)
        up.live("telemetry", tel.model_dump_json())
        for a in out.alerts:
            up.reliable("alerts", a.model_dump_json())
            sent += 1
        if sent >= 3:
            break
    assert sent == 3 and store.alerts(machine_id="HT-A") == [] and store.latest_telemetry("HT-A") is None
    up.set_online(True)
    assert len(store.alerts(machine_id="HT-A")) == 3
    assert not up.queue


def test_recent_telemetry_coach_feed_and_speed_limit(tmp_path):
    app = _app(tmp_path)
    ing = app.state.ingestor
    up = _Uplink(ing, "HT-B")
    sim = Simulation("nh_corridor", "typical", machine_id="HT-B", seed=0, trips=1, t0_wall=time.time())
    agent = CabAgent("HT-B", use_plan=False)
    for i, (tel, _) in enumerate(sim.stream()):
        out = agent.update(tel)
        up.live("telemetry", tel.model_dump_json())
        if out.coach:
            up.live("coach", out.coach.model_dump_json())
        if i > 400:
            break
    c = TestClient(app)
    rows = c.get("/telemetry/HT-B/recent?limit=5").json()
    assert 1 <= len(rows) <= 5 and rows[0]["ts"] <= rows[-1]["ts"] and rows[-1]["speed_limit_kmh"] > 0
    feed = c.get("/coach/HT-B?limit=3").json()
    assert len(feed) <= 3 and all(m["machine_id"] == "HT-B" for m in feed)
    assert c.get("/coach/NOBODY").json() == []
