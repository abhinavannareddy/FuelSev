"""One-command demo: the supervisor API plus a simulated fleet, with no broker or Docker.

Each truck runs the same simulator and in-cab agent as ``fuelsev machine``. Instead of going
through MQTT, its messages are handed to the API's ingestor, which applies the same topic,
size, schema and machine-ID checks. The dead zone is reproduced too: while a truck is
"out of coverage" its alerts and trips queue, and arrive, oldest first, when it is back.
"""

from __future__ import annotations

import logging
import shutil
import tempfile
import threading
import time
import webbrowser
from dataclasses import dataclass, field
from pathlib import Path

from .agent import CabAgent
from .api import Ingestor, Settings, create_app
from .drivers import FaultSpec
from .machine import load_model
from .mqtt import topic
from .schemas import Status
from .simulator import Simulation

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class DemoTruck:
    machine_id: str
    route: str
    driver: str
    lang: str
    seed: int = 0
    outages: tuple[tuple[float, float], ...] = ()
    faults: tuple[FaultSpec, ...] = ()


FLEET: tuple[DemoTruck, ...] = (
    DemoTruck("HT-01", "ghat_descent", "aggressive", "te", 0, outages=((900.0, 1500.0),),
              faults=(FaultSpec("overheat", 2400.0),)),
    DemoTruck("HT-02", "nh_corridor", "typical", "hi", 1),
    DemoTruck("HT-03", "quarry_haul", "eco", "en", 2),
    DemoTruck("HT-04", "urban_mixed", "aggressive", "hi", 3, faults=(FaultSpec("fuel_leak", 1500.0),)),
    DemoTruck("HT-05", "monsoon_site", "typical", "te", 4),
)


@dataclass
class _Uplink:
    """Delivers a truck's messages to the ingestor, queueing alerts and trips in a dead zone."""

    ingestor: Ingestor
    machine_id: str
    online: bool = True
    queue: list[tuple[str, str]] = field(default_factory=list)

    def _deliver(self, kind: str, payload: str) -> None:
        self.ingestor.handle(topic(self.machine_id, kind), payload.encode())

    def status(self, online: bool) -> None:
        self._deliver("status", Status(machine_id=self.machine_id, online=online, ts=time.time()).model_dump_json())

    def set_online(self, online: bool) -> None:
        if online == self.online:
            return
        self.online = online
        self.status(online)
        if online:
            for kind, payload in self.queue:
                self._deliver(kind, payload)
            self.queue.clear()

    def live(self, kind: str, payload: str) -> None:
        if self.online:  # telemetry and coach messages are a live view: lost while offline
            self._deliver(kind, payload)

    def reliable(self, kind: str, payload: str) -> None:
        if self.online:
            self._deliver(kind, payload)
        else:
            self.queue.append((kind, payload))


def _run_truck(truck: DemoTruck, ingestor: Ingestor, speedup: float, model_path: str | None, stop: threading.Event) -> None:
    link = _Uplink(ingestor, truck.machine_id)
    link.status(True)
    sim = Simulation(truck.route, truck.driver, machine_id=truck.machine_id, seed=truck.seed, trips=10**9,
                     faults=truck.faults, t0_wall=time.time())
    agent = CabAgent(truck.machine_id, anomaly_model=load_model(model_path), lang=truck.lang)
    period = 1.0 / speedup
    next_tick = time.monotonic()
    for tel, _truth in sim.stream():
        if stop.is_set():
            break
        tel = tel.model_copy(update={"ts": time.time()})
        link.set_online(not any(a <= tel.sim_t < b for a, b in truck.outages))
        out = agent.update(tel)
        link.live("telemetry", tel.model_dump_json())
        if out.coach:
            link.live("coach", out.coach.model_dump_json())
        for alert in out.alerts:
            link.reliable("alerts", alert.model_dump_json())
        if out.trip:
            link.reliable("trips", out.trip.model_dump_json())
        next_tick += period
        delay = next_tick - time.monotonic()
        if delay > 0:
            time.sleep(delay)
        else:
            next_tick = time.monotonic()
    link.status(False)


def run_demo(host: str = "127.0.0.1", port: int = 8000, speedup: float = 60.0, model_path: str | None = None,
             open_browser: bool = False, trucks: int = len(FLEET)) -> None:
    import uvicorn

    workdir = Path(tempfile.mkdtemp(prefix="fuelsev-demo-"))
    app = create_app(Settings(db_path=str(workdir / "demo.db"), ingest=False))
    stop = threading.Event()
    threads = [
        threading.Thread(target=_run_truck, args=(t, app.state.ingestor, speedup, model_path, stop), daemon=True, name=t.machine_id)
        for t in FLEET[:trucks]
    ]
    for th in threads:
        th.start()
    url = f"http://{host}:{port}"
    print(f"fuelsev demo: {len(threads)} simulated trucks at {speedup:g}x speed")
    print(f"  supervisor dashboard  {url}/")
    print(f"  in-cab display        {url}/cab?machine=HT-01")
    print(f"  API docs              {url}/docs")
    if model_path is None:
        print("  (no anomaly model given: rules only. Run `fuelsev train`, then add --model models/anomaly.joblib)")
    if open_browser:
        threading.Timer(1.5, lambda: webbrowser.open(url)).start()
    try:
        uvicorn.run(app, host=host, port=port, log_level="warning")
    finally:
        stop.set()
        shutil.rmtree(workdir, ignore_errors=True)
