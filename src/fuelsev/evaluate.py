"""Benchmark: fuel and CO2e per tonne for each driving style on every route.

Each cell is one or more simulated haul cycles (load, loaded leg, dump, empty return) run
through the same in-cab agent that runs on the machine, so the numbers come from the trip
summaries a supervisor would receive. Results are simulated, not measured on a real truck.
"""

from __future__ import annotations

import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from .agent import CabAgent
from .metrics import trip_kpis
from .routes import SCENARIOS
from .simulator import Simulation

DRIVERS = ("planner", "eco", "typical", "aggressive")
SEEDS = (1, 2, 3)


def _one(args: tuple[str, str, int]) -> dict:
    route, driver, seed = args
    sim = Simulation(route, driver, seed=seed, trips=1, machine_id="BENCH-1")
    agent = CabAgent("BENCH-1", use_plan=False)
    for tel, _ in sim.stream():
        agent.update(tel)
    trip = agent.finish()
    k = trip_kpis(trip)
    k.update(duration_min=trip.duration_s / 60.0, eco_score=trip.eco_score,
             harsh=trip.harsh_brake_events, overspeed_s=trip.overspeed_s,
             co2e_kg_per_tonne=k["co2e_kg_per_tonne"])
    return {"route": route, "driver": driver, "seed": seed, **k}


def run_benchmark(seeds=SEEDS, routes=None, drivers=DRIVERS, workers: int | None = None) -> dict:
    jobs = [(r, d, s) for r in (routes or list(SCENARIOS)) for d in drivers for s in seeds]
    with ProcessPoolExecutor(max_workers=workers) as ex:
        rows = list(ex.map(_one, jobs))
    cells: dict[str, dict[str, dict]] = {}
    for r in sorted({x["route"] for x in rows}):
        cells[r] = {}
        for d in drivers:
            sel = [x for x in rows if x["route"] == r and x["driver"] == d]
            fuel, pay = sum(x["fuel_l"] for x in sel), sum(x["payload_t"] for x in sel)
            cells[r][d] = {
                "l_per_tonne": round(fuel / pay, 3),
                "co2e_kg_per_tonne": round(fuel / pay * sel[0]["co2e_kg"] / sel[0]["fuel_l"], 3),
                "duration_min": round(float(np.mean([x["duration_min"] for x in sel])), 1),
                "fuel_cost_inr_per_tonne": round(fuel / pay * sel[0]["fuel_cost_inr"] / sel[0]["fuel_l"], 2),
                "eco_score": round(float(np.mean([x["eco_score"] for x in sel])), 1),
            }
    return {"seeds": list(seeds), "cells": cells, "rows": rows}


def to_markdown(bench: dict) -> str:
    drivers = list(next(iter(bench["cells"].values())))
    lines = ["| Route | " + " | ".join(f"{d} (L/t)" for d in drivers) + " | typical → planner |",
             "|---|" + "---|" * (len(drivers) + 1)]
    for r, c in bench["cells"].items():
        saving = 100 * (1 - c["planner"]["l_per_tonne"] / c["typical"]["l_per_tonne"])
        lines.append(f"| {r} | " + " | ".join(f"{c[d]['l_per_tonne']:.3f}" for d in drivers) + f" | {saving:.0f}% less fuel |")
    return "\n".join(lines)


def save(bench: dict, out_dir: str | Path) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "benchmark.json").write_text(json.dumps(bench, indent=1), encoding="utf-8")
    (out / "benchmark.md").write_text(to_markdown(bench) + "\n", encoding="utf-8")
