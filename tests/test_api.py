import time

import pytest
from fastapi.testclient import TestClient

from fuelsev.agent import CabAgent
from fuelsev.api import Ingestor, Settings, create_app
from fuelsev.mqtt import BrokerConfig
from fuelsev.simulator import Simulation


def _fill(store, machine="TRK-1", driver="aggressive", seed=1):
    sim = Simulation("urban_mixed", driver, machine_id=machine, seed=seed, trips=1, t0_wall=time.time() - 3500)
    ag = CabAgent(machine, use_plan=False)
    for tel, _ in sim.stream():
        out = ag.update(tel)
        store.add_telemetry(tel)
        for a in out.alerts:
            store.add_alert(a)
    trip = ag.finish()
    store.add_trip(trip)
    return trip


@pytest.fixture()
def client(tmp_path):
    app = create_app(Settings(db_path=str(tmp_path / "t.db"), ingest=False))
    trip = _fill(app.state.store)
    c = TestClient(app)
    c.trip = trip
    return c


def test_health_is_open_and_reports_version(client):
    assert client.get("/health").json()["status"] == "ok"


def test_report_totals_match_trip_even_though_telemetry_is_downsampled(client):
    rep = client.get("/report?hours=2").json()
    m = rep["machines"][0]
    assert m["window"]["fuel_l"] == pytest.approx(client.trip.fuel_l, rel=0.02)
    assert m["trips"]["l_per_tonne"] == pytest.approx(client.trip.fuel_l / client.trip.payload_t, rel=1e-3)
    assert rep["fleet"]["co2e_kg_per_tonne"] == pytest.approx(2.68 * client.trip.fuel_l / client.trip.payload_t, rel=1e-3)
    assert rep["emission_factor_kg_co2_per_l"] == pytest.approx(2.68)


def test_alert_filters_and_ack(client):
    allr = client.get("/alerts").json()
    assert allr
    crit = client.get("/alerts?severity=critical").json()
    assert all(a["severity"] == "critical" for a in crit)
    assert client.get("/alerts?severity=bogus").status_code == 422
    aid = allr[0]["id"]
    assert client.post(f"/alerts/{aid}/ack", json={"by": "ravi"}).json()["acknowledged"] is True
    assert client.post(f"/alerts/{aid}/ack").status_code == 404
    assert aid not in [a["id"] for a in client.get("/alerts?acknowledged=false").json()]


def test_latest_and_404(client):
    assert client.get("/telemetry/TRK-1/latest").json()["machine_id"] == "TRK-1"
    assert client.get("/telemetry/NOPE/latest").status_code == 404


def test_csv_report_and_injection_guard(client):
    r = client.get("/report.csv?hours=2")
    assert r.headers["content-type"].startswith("text/csv")
    assert r.text.splitlines()[1].startswith("TRK-1,")
    from fuelsev.api import _safe
    assert _safe("=HYPERLINK(1)") == "'=HYPERLINK(1)"
    assert _safe("-2+3") == "'-2+3"
    assert _safe(None) == "" and _safe(3.5) == 3.5


def test_api_key_is_enforced(tmp_path):
    app = create_app(Settings(db_path=str(tmp_path / "k.db"), ingest=False, api_key="s3cret"))
    c = TestClient(app)
    assert c.get("/health").status_code == 200
    assert c.get("/machines").status_code == 401
    assert c.get("/machines", headers={"X-API-Key": "wrong"}).status_code == 401
    assert c.get("/machines", headers={"X-API-Key": "s3cret"}).status_code == 200


def test_ingestor_rejects_spoofed_and_malformed_messages(tmp_path):
    app = create_app(Settings(db_path=str(tmp_path / "i.db"), ingest=False))
    ing = Ingestor(app.state.store, BrokerConfig())
    sim = Simulation("nh_corridor", "typical", machine_id="HT-A", seed=0, trips=1)
    tel = next(iter(sim.stream()))[0]
    good = tel.model_dump_json().encode()
    ing.handle("fuelsev/v1/HT-A/telemetry", good)
    assert ing.accepted == 1
    ing.handle("fuelsev/v1/HT-B/telemetry", good)           # payload names another machine
    ing.handle("fuelsev/v1/HT-A/telemetry", b"not json")     # malformed
    ing.handle("other/topic", good)                          # foreign topic
    ing.handle("fuelsev/v1/HT-A/telemetry", b"x" * 70000)   # oversized
    assert ing.accepted == 1 and ing.rejected == 4
    assert [m["machine_id"] for m in app.state.store.machines()] == ["HT-A"]


def test_duplicate_alert_delivery_is_stored_once(tmp_path):
    app = create_app(Settings(db_path=str(tmp_path / "d.db"), ingest=False))
    store = app.state.store
    ing = Ingestor(store, BrokerConfig())
    a = None
    sim = Simulation("urban_mixed", "aggressive", machine_id="HT-A", seed=1, trips=1)
    ag = CabAgent("HT-A", use_plan=False)
    for tel, _ in sim.stream():
        out = ag.update(tel)
        if out.alerts:
            a = out.alerts[0]
            break
    ing.handle("fuelsev/v1/HT-A/alerts", a.model_dump_json().encode())
    ing.handle("fuelsev/v1/HT-A/alerts", a.model_dump_json().encode())
    assert len(store.alerts(machine_id="HT-A")) == 1 and ing.duplicates == 1
