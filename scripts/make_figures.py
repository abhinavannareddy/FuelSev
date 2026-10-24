"""Regenerate the figures in docs/img from the project's own code and results/*.json.

    pip install -e ".[plots]"
    python scripts/make_figures.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from fuelsev.fuelmap import load_default_fuel_map
from fuelsev.planner import plan_leg
from fuelsev.routes import SCENARIOS
from fuelsev.vehicle import Vehicle

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "img"
BG, CARD, LINE, TEXT, MUTED = "#0b1219", "#111b25", "#223140", "#e8eef4", "#8a9bad"
GREEN, AMBER, RED, BLUE, VIOLET = "#3ddc84", "#f6b93b", "#ff6b6b", "#4da3ff", "#a78bfa"
STYLE = {"planner": GREEN, "eco": "#7bd88f", "typical": AMBER, "aggressive": RED}

plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": CARD, "savefig.facecolor": BG, "text.color": TEXT,
    "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED, "axes.edgecolor": LINE,
    "axes.grid": True, "grid.color": LINE, "grid.linewidth": 0.7, "font.family": "DejaVu Sans",
    "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
})


def _save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / name, dpi=170, bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)
    print("wrote", OUT / name)


def benchmark():
    cells = json.loads((ROOT / "results" / "benchmark.json").read_text())["cells"]
    routes = list(cells)
    drivers = ["aggressive", "typical", "eco", "planner"]
    fig, ax = plt.subplots(figsize=(11, 4.9))
    w = 0.2
    x = np.arange(len(routes))
    for k, d in enumerate(drivers):
        vals = [cells[r][d]["l_per_tonne"] for r in routes]
        ax.bar(x + (k - 1.5) * w, vals, w * 0.9, label=d, color=STYLE[d], zorder=3)
    for i, r in enumerate(routes):
        t, p = cells[r]["typical"]["l_per_tonne"], cells[r]["planner"]["l_per_tonne"]
        ax.text(i, max(cells[r][d]["l_per_tonne"] for d in drivers) + 0.025, f"-{100 * (1 - p / t):.0f}%",
                ha="center", color=GREEN, fontweight="bold", fontsize=11)
    ax.set_xticks(x, [r.replace("_", " ") for r in routes])
    ax.set_ylabel("fuel per tonne hauled (L/t)")
    ax.set_ylim(0, 0.95)
    ax.grid(axis="x", visible=False)
    ax.set_title("Fuel per tonne by driving style  ·  green label = typical → planner", loc="left", color=TEXT, fontsize=13, fontweight="bold", pad=14)
    ax.legend(ncol=4, loc="upper right", labelcolor=TEXT)
    _save(fig, "benchmark.png")


def planner():
    veh, scn = Vehicle(), SCENARIOS["ghat_climb"]
    prof = scn.leg("loaded").profile()
    plan = plan_leg(veh, prof, 22_000.0)
    s_km = plan.s / 1000.0
    elev = np.concatenate([[0], np.cumsum(prof.grade_pct / 100.0 * np.diff(prof.s))])
    fig, (a, b, c) = plt.subplots(3, 1, figsize=(11, 7.4), sharex=True, gridspec_kw={"height_ratios": [1, 1.6, 0.8], "hspace": 0.1})
    a.fill_between(s_km, elev, elev.min() - 10, color=VIOLET, alpha=0.25, zorder=2)
    a.plot(s_km, elev, color=VIOLET, lw=2, zorder=3)
    a.set_ylabel("elevation (m)")
    a.set_title("Dynamic-programming plan, ghat climb, 22 t payload", loc="left", color=TEXT, fontsize=13, fontweight="bold", pad=12)
    b.step(s_km, prof.cap_ms * 3.6, where="post", color=MUTED, lw=1.4, ls="--", label="speed cap (law and traffic)")
    b.plot(s_km, plan.v * 3.6, color=GREEN, lw=2.4, label="planned speed", zorder=3)
    b.set_ylabel("speed (km/h)")
    b.set_ylim(0, None)
    b.legend(loc="upper right", labelcolor=TEXT)
    g = np.append(plan.gear, plan.gear[-1])
    c.step(s_km, g, where="post", color=BLUE, lw=2)
    c.set_ylabel("gear")
    c.set_xlabel("distance (km)")
    c.set_yticks(sorted(set(g.astype(int)))[::2])
    _save(fig, "planner.png")


def fuel_map():
    fm = load_default_fuel_map()
    rpm = np.linspace(fm.rpm[0], fm.rpm[-1], 160)
    tq = np.linspace(fm.torque_nm[0], fm.torque_nm[-1], 160)
    R, T = np.meshgrid(rpm, tq)
    bsfc = np.clip(fm.bsfc_gkwh(R, T), 190, 320)
    fig, ax = plt.subplots(figsize=(8.2, 5.4))
    cs = ax.contourf(R, T, bsfc, levels=np.linspace(190, 320, 27), cmap="viridis_r", zorder=1)
    ax.contour(R, T, bsfc, levels=[200, 205, 215, 235, 270], colors="white", linewidths=0.6, alpha=0.5)
    ax.grid(False)
    cb = fig.colorbar(cs, ax=ax, pad=0.02)
    cb.set_label("g/kWh  (lower is better)", color=MUTED)
    cb.outline.set_edgecolor(LINE)
    i = np.unravel_index(np.nanargmin(fm.bsfc_gkwh(R, T)), R.shape)
    ax.scatter(R[i], T[i], s=140, color=GREEN, edgecolor="white", zorder=5)
    ax.annotate(f"sweet spot\n{fm.bsfc_gkwh(R[i], T[i]):.0f} g/kWh", (R[i], T[i]), (R[i] + 180, T[i] - 380), color=TEXT, fontweight="bold",
                arrowprops={"arrowstyle": "-", "color": "white"})
    ax.set_xlabel("engine speed (rpm)")
    ax.set_ylabel("torque (N·m)")
    ax.set_title("Synthetic engine fuel map: why gear choice matters", loc="left", color=TEXT, fontsize=13, fontweight="bold", pad=12)
    _save(fig, "fuel-map.png")


def detection():
    d = json.loads((ROOT / "results" / "detection.json").read_text())
    names = [("rules", "Rules"), ("forest", "Isolation Forest"), ("union", "Rules or Forest")]
    metrics = ["precision", "recall", "f1"]
    cols = [BLUE, GREEN, VIOLET]
    fig, ax = plt.subplots(figsize=(9.5, 4.4))
    x = np.arange(len(names))
    w = 0.26
    for k, m in enumerate(metrics):
        v = [d[n][m] for n, _ in names]
        ax.bar(x + (k - 1) * w, v, w * 0.9, color=cols[k], label=m, zorder=3)
        for xi, vi in zip(x + (k - 1) * w, v, strict=True):
            ax.text(xi, vi + 0.015, f"{vi:.2f}", ha="center", fontsize=9, color=TEXT)
    ax.set_xticks(x, [f"{lab}\nfalse alarms {d[n]['false_alarm_rate'] * 100:.1f}%" for n, lab in names])
    ax.set_ylim(0, 1)
    ax.grid(axis="x", visible=False)
    ax.legend(ncol=3, loc="upper right", labelcolor=TEXT)
    ax.set_title(f"Unsafe-driving detection on {d['n_windows']:,} held-out windows  ·  forest AUC {d['auc_forest']:.2f}",
                 loc="left", color=TEXT, fontsize=13, fontweight="bold", pad=12)
    _save(fig, "detection.png")


if __name__ == "__main__":
    benchmark()
    planner()
    fuel_map()
    detection()
