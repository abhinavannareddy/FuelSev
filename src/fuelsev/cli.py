"""Command line entry point: ``fuelsev <command>``."""

from __future__ import annotations

import argparse
import json
import logging
import sys
import threading
from pathlib import Path

from .drivers import FAULT_KINDS, PROFILES, FaultSpec
from .routes import SCENARIOS


def _faults(items: list[str] | None) -> list[FaultSpec]:
    out = []
    for item in items or []:
        kind, _, at = item.partition("@")
        out.append(FaultSpec(kind, float(at or 600)))
    return out


def _outages(items: list[str] | None) -> list[tuple[float, float]]:
    out = []
    for item in items or []:
        a, _, b = item.partition("-")
        out.append((float(a), float(b)))
    return out


def cmd_build_fuelmap(args) -> int:
    from .fuelmap import build_synthetic_fuel_map

    build_synthetic_fuel_map().to_csv(args.out)
    print(f"wrote {args.out}")
    return 0


def cmd_plan(args) -> int:
    from .planner import plan_leg
    from .vehicle import Vehicle

    scn = SCENARIOS[args.route]
    plan = plan_leg(Vehicle(), scn.leg(args.leg).profile(), args.payload_t * 1000.0)
    print(json.dumps(plan.summary(), indent=2))
    return 0


def cmd_simulate(args) -> int:
    from .agent import CabAgent
    from .metrics import trip_kpis
    from .simulator import Simulation

    sim = Simulation(args.route, args.driver, seed=args.seed, trips=args.trips, faults=tuple(_faults(args.fault)))
    agent = CabAgent("SIM-001")
    alerts, trips = [], []
    for tel, _ in sim.stream():
        out = agent.update(tel)
        alerts += out.alerts
        if out.trip:
            trips.append(out.trip)
    trips.append(agent.finish())
    for t in trips:
        print(json.dumps({"trip": t.trip_no, **{k: (round(v, 3) if isinstance(v, float) else v) for k, v in trip_kpis(t).items()},
                          "eco_score": t.eco_score, "harsh_brakes": t.harsh_brake_events}))
    print(f"{len(alerts)} alerts:", ", ".join(sorted({a.code for a in alerts})) or "none")
    return 0


def cmd_train(args) -> int:
    from .training import train_default

    model, n = train_default(args.false_alarm_rate)
    model.save(args.out)
    print(f"trained on {n} windows; saved {args.out}")
    return 0


def cmd_evaluate(args) -> int:
    from . import evaluate as ev
    from .anomaly import AnomalyModel
    from .training import evaluate_detection

    bench = ev.run_benchmark(seeds=tuple(range(1, args.seeds + 1)))
    ev.save(bench, args.out)
    print(ev.to_markdown(bench))
    if args.model and Path(args.model).exists():
        det = evaluate_detection(AnomalyModel.load(args.model))
        (Path(args.out) / "detection.json").write_text(json.dumps(det.__dict__, indent=1), encoding="utf-8")
        print(json.dumps(det.__dict__, indent=1))
    return 0


def cmd_machine(args) -> int:
    from .machine import MachineConfig, install_signal_handlers, run_machine

    stop = threading.Event()
    install_signal_handlers(stop)
    cfg = MachineConfig(
        machine_id=args.id, route=args.route, driver=args.driver, trips=args.trips, speedup=args.speedup,
        lang=args.lang, seed=args.seed, model_path=args.model, outages=_outages(args.outage), faults=_faults(args.fault),
    )
    print(json.dumps(run_machine(cfg, stop)))
    return 0


def cmd_api(args) -> int:
    import uvicorn

    uvicorn.run("fuelsev.api:app_factory", factory=True, host=args.host, port=args.port, log_level="info")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="fuelsev", description="In-cab fuel coaching and supervisor alerts for haul trucks.")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("build-fuelmap", help="regenerate the synthetic fuel map CSV")
    s.add_argument("--out", default="fuel_map.csv")
    s.set_defaults(fn=cmd_build_fuelmap)

    s = sub.add_parser("plan", help="run the dynamic-programming planner on one leg")
    s.add_argument("--route", choices=sorted(SCENARIOS), default="nh_corridor")
    s.add_argument("--leg", choices=["loaded", "empty"], default="loaded")
    s.add_argument("--payload-t", type=float, default=22.0)
    s.set_defaults(fn=cmd_plan)

    s = sub.add_parser("simulate", help="simulate trips locally and print trip KPIs")
    s.add_argument("--route", choices=sorted(SCENARIOS), default="nh_corridor")
    s.add_argument("--driver", choices=sorted(PROFILES), default="typical")
    s.add_argument("--trips", type=int, default=1)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--fault", action="append", metavar="KIND@SECONDS", help=f"inject a fault ({', '.join(FAULT_KINDS)})")
    s.set_defaults(fn=cmd_simulate)

    s = sub.add_parser("train", help="train the Isolation Forest on simulated baseline driving")
    s.add_argument("--out", default="models/anomaly.joblib")
    s.add_argument("--false-alarm-rate", type=float, default=0.02)
    s.set_defaults(fn=cmd_train)

    s = sub.add_parser("evaluate", help="benchmark fuel per tonne and detection quality")
    s.add_argument("--out", default="results")
    s.add_argument("--seeds", type=int, default=3)
    s.add_argument("--model", default="models/anomaly.joblib")
    s.set_defaults(fn=cmd_evaluate)

    s = sub.add_parser("machine", help="run a simulated machine that publishes over MQTT")
    s.add_argument("--id", default="HT-01")
    s.add_argument("--route", choices=sorted(SCENARIOS), default="nh_corridor")
    s.add_argument("--driver", choices=sorted(PROFILES), default="typical")
    s.add_argument("--trips", type=int, default=0, help="0 = run until stopped")
    s.add_argument("--speedup", type=float, default=20.0)
    s.add_argument("--lang", choices=["en", "hi", "te"], default="en")
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--model", default=None)
    s.add_argument("--outage", action="append", metavar="START-END", help="cut the link between two machine-clock seconds")
    s.add_argument("--fault", action="append", metavar="KIND@SECONDS")
    s.set_defaults(fn=cmd_machine)

    s = sub.add_parser("api", help="run the supervisor API (and MQTT ingestor)")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8000)
    s.set_defaults(fn=cmd_api)
    return p


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    args = build_parser().parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
