"""Supervisor side: an MQTT ingestor that fills the store, and a FastAPI app to read it.

Configuration comes from the environment so the same image runs anywhere:

    FUELSEV_DB_PATH      SQLite file (default ./fuelsev.db)
    FUELSEV_API_KEY      if set, every endpoint except /health needs the header X-API-Key
    FUELSEV_MQTT_*       broker host, port, user, password, TLS (see mqtt.BrokerConfig)
    FUELSEV_INGEST       set to 0 to run the API without subscribing (used in tests)
"""

from __future__ import annotations

import csv
import hmac
import io
import logging
import os
import threading
import time
from collections import deque
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path

from fastapi import Depends, FastAPI, Header, HTTPException, Query
from fastapi.responses import HTMLResponse, Response
from pydantic import BaseModel, ValidationError

from . import __version__
from .constants import CO2_KG_PER_L_DIESEL
from .mqtt import MAX_PAYLOAD_BYTES, TOPIC_ROOT, BrokerConfig, Link, parse_topic
from .schemas import Alert, CoachMessage, Status, Telemetry, TripSummary
from .store import Store

log = logging.getLogger(__name__)

_CSV_DANGEROUS = ("=", "+", "-", "@", "\t", "\r")

_WEB = Path(__file__).parent / "web"
COACH_KEEP = 25  # coach messages remembered per machine (a live view, never stored)


@dataclass
class Settings:
    db_path: str = "fuelsev.db"
    api_key: str | None = None
    ingest: bool = True
    broker: BrokerConfig = field(default_factory=BrokerConfig.from_env)

    @classmethod
    def from_env(cls) -> Settings:
        return cls(
            db_path=os.environ.get("FUELSEV_DB_PATH", "fuelsev.db"),
            api_key=os.environ.get("FUELSEV_API_KEY") or None,
            ingest=os.environ.get("FUELSEV_INGEST", "1") != "0",
            broker=BrokerConfig.from_env(),
        )


class Ingestor:
    """Subscribes to every machine and writes valid messages to the store.

    A message is accepted only if its topic is well formed, the payload is small, parses
    against the schema, and names the same machine as the topic. Anything else is counted
    and dropped, so one misbehaving device cannot write data under another machine's name.
    """

    def __init__(self, store: Store, broker: BrokerConfig):
        self.store = store
        self.accepted = 0
        self.rejected = 0
        self.duplicates = 0
        self.coach: dict[str, deque[dict]] = {}
        self._lock = threading.Lock()
        self.link = Link(
            broker,
            client_id="fuelsev-api",
            on_message=self.handle,
            subscriptions=((f"{TOPIC_ROOT}/+/telemetry", 0), (f"{TOPIC_ROOT}/+/alerts", 1),
                           (f"{TOPIC_ROOT}/+/trips", 1), (f"{TOPIC_ROOT}/+/status", 1),
                           (f"{TOPIC_ROOT}/+/coach", 0)),
            persistent_session=True,
        )

    def start(self) -> None:
        self.link.start()

    def stop(self) -> None:
        self.link.stop()

    def _count(self, attr: str) -> None:
        with self._lock:
            setattr(self, attr, getattr(self, attr) + 1)

    def handle(self, topic_name: str, payload: bytes) -> None:
        parsed = parse_topic(topic_name)
        if parsed is None or len(payload) > MAX_PAYLOAD_BYTES:
            self._count("rejected")
            return
        machine_id, kind = parsed
        try:
            if kind == "telemetry":
                msg = Telemetry.model_validate_json(payload)
            elif kind == "alerts":
                msg = Alert.model_validate_json(payload)
            elif kind == "trips":
                msg = TripSummary.model_validate_json(payload)
            elif kind == "status":
                msg = Status.model_validate_json(payload)
            else:
                msg = CoachMessage.model_validate_json(payload)
        except ValidationError:
            self._count("rejected")
            return
        if msg.machine_id != machine_id:
            self._count("rejected")
            return
        if kind == "coach":  # shown in the live cab view; not stored
            with self._lock:
                self.coach.setdefault(machine_id, deque(maxlen=COACH_KEEP)).append(msg.model_dump())
            self._count("accepted")
            return
        try:
            if kind == "telemetry":
                self.store.add_telemetry(msg)
            elif kind == "alerts":
                if not self.store.add_alert(msg):
                    self._count("duplicates")
                    return
            elif kind == "trips":
                if not self.store.add_trip(msg):
                    self._count("duplicates")
                    return
            else:
                self.store.set_status(msg)
        except Exception:  # a storage error must not kill the MQTT thread
            log.exception("failed to store %s message from %s", kind, machine_id)
            self._count("rejected")
            return
        self._count("accepted")


# ---------------------------------------------------------------------- app
def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings.from_env()
    store = Store(settings.db_path)
    ingestor = Ingestor(store, settings.broker)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if settings.ingest:
            ingestor.start()
        try:
            yield
        finally:
            if settings.ingest:
                ingestor.stop()

    app = FastAPI(
        title="fuelsev supervisor API",
        version=__version__,
        description="Fuel and safety monitoring for haul trucks: alerts, trips and fuel/CO2e reports.",
        lifespan=lifespan,
    )
    app.state.store = store
    app.state.ingestor = ingestor

    def require_key(x_api_key: str | None = Header(default=None)) -> None:
        if settings.api_key is None:
            return
        if x_api_key is None or not hmac.compare_digest(x_api_key, settings.api_key):
            raise HTTPException(status_code=401, detail="missing or invalid API key")

    auth = [Depends(require_key)]

    class AckBody(BaseModel):
        by: str = "supervisor"

    @app.get("/health")
    def health() -> dict:
        ing = app.state.ingestor
        return {
            "status": "ok",
            "version": __version__,
            "mqtt_connected": bool(settings.ingest and ing.link.connected),
            "auth_required": settings.api_key is not None,
            "accepted": ing.accepted,
            "rejected": ing.rejected,
            "duplicates": ing.duplicates,
        }

    def _page(name: str) -> HTMLResponse:
        return HTMLResponse(
            (_WEB / name).read_text(encoding="utf-8"),
            headers={"Content-Security-Policy": "default-src 'self'; style-src 'unsafe-inline'; script-src 'unsafe-inline'; "
                     "connect-src 'self'; img-src 'self' data:", "X-Content-Type-Options": "nosniff", "Cache-Control": "no-cache"},
        )

    @app.get("/", include_in_schema=False)
    def dashboard() -> HTMLResponse:
        return _page("dashboard.html")

    @app.get("/cab", include_in_schema=False)
    def cab_display() -> HTMLResponse:
        return _page("cab.html")

    @app.get("/machines", dependencies=auth)
    def machines() -> list[dict]:
        return store.machines()

    @app.get("/telemetry/{machine_id}/latest", dependencies=auth)
    def latest(machine_id: str) -> dict:
        row = store.latest_telemetry(machine_id)
        if row is None:
            raise HTTPException(status_code=404, detail="no telemetry for this machine")
        return row

    @app.get("/telemetry/{machine_id}/recent", dependencies=auth)
    def recent(machine_id: str, limit: int = Query(default=120, ge=1, le=1000)) -> list[dict]:
        return store.recent_telemetry(machine_id, limit)

    @app.get("/coach/{machine_id}", dependencies=auth)
    def coach_feed(machine_id: str, limit: int = Query(default=10, ge=1, le=COACH_KEEP)) -> list[dict]:
        """The latest cab-display messages, newest first (kept in memory only)."""
        with ingestor._lock:
            items = list(ingestor.coach.get(machine_id, ()))
        return items[::-1][:limit]

    @app.get("/alerts", dependencies=auth)
    def alerts(
        machine_id: str | None = None,
        severity: str | None = Query(default=None, pattern="^(info|warning|critical)$"),
        code: str | None = None,
        acknowledged: bool | None = None,
        since: float | None = None,
        limit: int = Query(default=100, ge=1, le=1000),
        offset: int = Query(default=0, ge=0),
    ) -> list[dict]:
        return store.alerts(machine_id, severity, code, acknowledged, since, limit, offset)

    @app.post("/alerts/{alert_id}/ack", dependencies=auth)
    def ack(alert_id: str, body: AckBody | None = None) -> dict:
        by = (body.by if body else "supervisor")[:64]
        if not store.acknowledge(alert_id, by):
            raise HTTPException(status_code=404, detail="alert not found or already acknowledged")
        return {"id": alert_id, "acknowledged": True, "by": by}

    @app.get("/trips", dependencies=auth)
    def trips(
        machine_id: str | None = None,
        since: float | None = None,
        until: float | None = None,
        limit: int = Query(default=100, ge=1, le=1000),
    ) -> list[dict]:
        return store.trips(machine_id, since, until, limit)

    def _window(since: float | None, until: float | None, hours: float) -> tuple[float, float]:
        end = until if until is not None else time.time()
        start = since if since is not None else end - hours * 3600.0
        if start > end:
            raise HTTPException(status_code=422, detail="'since' is after 'until'")
        return start, end

    @app.get("/report", dependencies=auth)
    def report(
        machine_id: str | None = None,
        since: float | None = None,
        until: float | None = None,
        hours: float = Query(default=24.0, gt=0, le=24 * 90),
    ) -> dict:
        start, end = _window(since, until, hours)
        out = store.report(start, end, machine_id)
        out["emission_factor_kg_co2_per_l"] = CO2_KG_PER_L_DIESEL
        return out

    @app.get("/report.csv", dependencies=auth)
    def report_csv(
        machine_id: str | None = None,
        since: float | None = None,
        until: float | None = None,
        hours: float = Query(default=24.0, gt=0, le=24 * 90),
    ) -> Response:
        start, end = _window(since, until, hours)
        data = store.report(start, end, machine_id)
        buf = io.StringIO()
        w = csv.writer(buf)
        w.writerow(["machine_id", "trips", "payload_t", "trip_fuel_l", "l_per_tonne", "co2e_kg_per_tonne",
                    "window_fuel_l", "window_distance_km", "window_co2e_kg", "idle_share", "avg_eco_score",
                    "harsh_brake_events", "alerts"])
        for m in data["machines"]:
            t, win = m["trips"], m["window"]
            w.writerow([_safe(v) for v in (
                m["machine_id"], t["trips"], t["payload_t"], t["fuel_l"], t["l_per_tonne"], t["co2e_kg_per_tonne"],
                win["fuel_l"], win["distance_km"], win["co2e_kg"], win["idle_share"], t["avg_eco_score"],
                t["harsh_brake_events"], sum(m["alerts"].values()),
            )])
        return Response(
            buf.getvalue(),
            media_type="text/csv",
            headers={"Content-Disposition": 'attachment; filename="fuelsev_report.csv"'},
        )

    return app


def _safe(v: object) -> object:
    """Neutralise spreadsheet formulas in text cells (CSV injection)."""
    if isinstance(v, str) and v.startswith(_CSV_DANGEROUS):
        return "'" + v
    return "" if v is None else v


def app_factory() -> FastAPI:
    """Entry point for ``uvicorn fuelsev.api:app_factory --factory``."""
    return create_app()
