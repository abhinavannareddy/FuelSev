"""The machine process: simulated sensors, the in-cab agent, and the MQTT uplink.

On a real machine the telemetry would come from the CAN bus; here the simulator stands in
for it. Everything downstream (agent, MQTT topics, buffering) is what would run in the cab.
"""

from __future__ import annotations

import logging
import signal
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

from .agent import CabAgent
from .anomaly import AnomalyModel
from .drivers import FaultSpec
from .mqtt import BrokerConfig, Link, ReliableSender, topic
from .schemas import Status
from .simulator import Simulation

log = logging.getLogger(__name__)


@dataclass
class MachineConfig:
    machine_id: str
    route: str = "nh_corridor"
    driver: str = "typical"
    trips: int = 0  # 0 means run until stopped
    speedup: float = 20.0  # simulated seconds per wall-clock second; 0 = as fast as possible
    lang: str = "en"
    seed: int = 0
    model_path: str | None = None
    outages: list[tuple[float, float]] = field(default_factory=list)  # (start_s, end_s) of machine time
    faults: list[FaultSpec] = field(default_factory=list)
    broker: BrokerConfig = field(default_factory=BrokerConfig.from_env)

    def in_outage(self, sim_t: float) -> bool:
        return any(a <= sim_t < b for a, b in self.outages)


def load_model(path: str | None) -> AnomalyModel | None:
    if not path:
        return None
    p = Path(path)
    if not p.exists():
        log.warning("anomaly model %s not found; running with rules only", p)
        return None
    return AnomalyModel.load(p)


def run_machine(cfg: MachineConfig, stop: threading.Event | None = None) -> dict:
    """Run until the trips are done or ``stop`` is set. Returns simple counters."""
    stop = stop or threading.Event()
    mid = cfg.machine_id

    def status_json(online: bool) -> str:
        return Status(machine_id=mid, online=online, ts=time.time()).model_dump_json()

    link = Link(
        cfg.broker,
        client_id=f"fuelsev-machine-{mid}",
        will=(topic(mid, "status"), status_json(False)),
        persistent_session=True,
    )
    link.start()
    if not link.wait_connected(15.0):
        log.warning("broker not reachable yet; alerts will queue on the machine until it is")

    sender = ReliableSender(lambda t, p, q, r: link.publish(t, p, qos=q, retain=r))
    sender.send(topic(mid, "status"), status_json(True), qos=1, retain=True)

    sim = Simulation(
        cfg.route,
        cfg.driver,
        machine_id=mid,
        seed=cfg.seed,
        trips=cfg.trips if cfg.trips > 0 else 10**9,
        faults=tuple(cfg.faults),
        t0_wall=time.time(),
    )
    agent = CabAgent(mid, anomaly_model=load_model(cfg.model_path), lang=cfg.lang)

    counts = {"samples": 0, "alerts": 0, "trips": 0, "coach": 0}
    period = 1.0 / cfg.speedup if cfg.speedup > 0 else 0.0
    next_tick = time.monotonic()
    completed = False
    try:
        for tel, _truth in sim.stream():
            if stop.is_set():
                break
            tel = tel.model_copy(update={"ts": time.time()})  # stamp with the real clock
            link.simulated_outage = cfg.in_outage(tel.sim_t)
            out = agent.update(tel)
            counts["samples"] += 1

            link.publish(topic(mid, "telemetry"), tel.model_dump_json(), qos=0)
            if out.coach:
                counts["coach"] += 1
                link.publish(topic(mid, "coach"), out.coach.model_dump_json(), qos=0)
            for alert in out.alerts:
                counts["alerts"] += 1
                sender.send(topic(mid, "alerts"), alert.model_dump_json(), qos=1)
            if out.trip:
                counts["trips"] += 1
                sender.send(topic(mid, "trips"), out.trip.model_dump_json(), qos=1)
            sender.flush()

            if period:
                next_tick += period
                delay = next_tick - time.monotonic()
                if delay > 0:
                    time.sleep(delay)
                else:
                    next_tick = time.monotonic()
        else:
            completed = True
    finally:
        # A trip cut short by a shutdown is not reported: it would distort fuel-per-tonne figures.
        final = agent.finish() if completed else None
        link.simulated_outage = False
        if final:
            counts["trips"] += 1
            sender.send(topic(mid, "trips"), final.model_dump_json(), qos=1)
        sender.send(topic(mid, "status"), status_json(False), qos=1, retain=True)
        deadline = time.monotonic() + 10.0
        while len(sender) and time.monotonic() < deadline:
            sender.flush()
            time.sleep(0.2)
        time.sleep(0.3)  # let paho finish in-flight QoS 1 acknowledgements
        link.stop()
    counts["dropped_from_queue"] = sender.dropped
    return counts


def install_signal_handlers(stop: threading.Event) -> None:
    def _handler(signum, frame):  # noqa: ARG001
        stop.set()

    signal.signal(signal.SIGINT, _handler)
    signal.signal(signal.SIGTERM, _handler)
