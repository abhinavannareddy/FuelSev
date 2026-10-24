"""Training data and detection evaluation for the unsafe-driving detector.

Training is unsupervised: the Isolation Forest sees windows from a baseline fleet of eco,
planner and typical drivers on every route, with no labels. Evaluation uses held-out seeds
and includes the aggressive driver; the simulator's hidden ``Truth.unsafe`` flag (hard
braking, tailgating near-misses, neutral coasting, overspeed episodes) provides labels.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import roc_auc_score

from .anomaly import AnomalyModel, train_anomaly_model
from .features import HOP_S, WINDOW_S, is_driving, session_windows, window_features
from .routes import SCENARIOS
from .rules import RuleEngine
from .simulator import Simulation
from .vehicle import Vehicle

BASELINE_DRIVERS = ("eco", "planner", "typical")
TRAIN_SEEDS = tuple(range(100, 103))
EVAL_SEEDS = tuple(range(200, 204))


def _session(route: str, driver: str, seed: int, veh: Vehicle):
    sim = Simulation(route, driver, vehicle=veh, seed=seed, trips=1, machine_id="SIM-TRAIN")
    res = sim.run()
    return res.telemetry, [t.unsafe for t in res.truth]


def build_training_set(seeds=TRAIN_SEEDS, routes=None, drivers=BASELINE_DRIVERS) -> np.ndarray:
    veh = Vehicle()
    xs = []
    for route in routes or list(SCENARIOS):
        for driver in drivers:
            for seed in seeds:
                tel, _ = _session(route, driver, seed, veh)
                X, _ = session_windows(tel)
                if len(X):
                    xs.append(X)
    return np.vstack(xs)


def train_default(false_alarm_rate: float = 0.02, seed: int = 0) -> tuple[AnomalyModel, int]:
    X = build_training_set()
    return train_anomaly_model(X, false_alarm_rate=false_alarm_rate, seed=seed), len(X)


@dataclass
class DetectionResult:
    n_windows: int
    n_unsafe: int
    rules: dict
    forest: dict
    union: dict
    auc_forest: float | None
    flagged_by_driver: dict


def _rule_flags(tel, veh: Vehicle) -> np.ndarray:
    """Per-sample flag: a safety rule fired at this second."""
    engine = RuleEngine(veh.fuel_map, veh.params.payload_rated_kg / 1000.0)
    flags = np.zeros(len(tel), dtype=bool)
    for i, t in enumerate(tel):
        flags[i] = any(d.category == "safety" for d in engine.update(t))
    return flags


def _prf(pred: np.ndarray, y: np.ndarray) -> dict:
    tp = int((pred & (y == 1)).sum())
    fp = int((pred & (y == 0)).sum())
    fn = int((~pred & (y == 1)).sum())
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    fpr = fp / max(int((y == 0).sum()), 1)
    return {"precision": round(p, 3), "recall": round(r, 3), "f1": round(f1, 3), "false_alarm_rate": round(fpr, 3)}


def evaluate_detection(model: AnomalyModel, seeds=EVAL_SEEDS, routes=None) -> DetectionResult:
    veh = Vehicle()
    X_all, y_all, r_all, d_all = [], [], [], []
    for route in routes or list(SCENARIOS):
        for driver in ("eco", "planner", "typical", "aggressive"):
            for seed in seeds:
                tel, unsafe = _session(route, driver, seed, veh)
                flags = _rule_flags(tel, veh)
                for start in range(0, len(tel) - WINDOW_S + 1, HOP_S):
                    w = tel[start : start + WINDOW_S]
                    if not is_driving(w):
                        continue
                    X_all.append(window_features(w))
                    y_all.append(int(np.mean(unsafe[start : start + WINDOW_S]) >= 0.5))
                    r_all.append(bool(flags[start : start + WINDOW_S].any()))
                    d_all.append(driver)
    X, y = np.vstack(X_all), np.asarray(y_all)
    rules = np.asarray(r_all)
    scores = model.scores(X)
    forest = scores < model.threshold
    drivers = np.asarray(d_all)
    auc = float(roc_auc_score(y, -scores)) if 0 < y.sum() < len(y) else None
    return DetectionResult(
        n_windows=len(y),
        n_unsafe=int(y.sum()),
        rules=_prf(rules, y),
        forest=_prf(forest, y),
        union=_prf(rules | forest, y),
        auc_forest=None if auc is None else round(auc, 3),
        flagged_by_driver={
            d: {
                "rules": round(float(rules[drivers == d].mean()), 3),
                "forest": round(float(forest[drivers == d].mean()), 3),
                "unsafe_share": round(float(y[drivers == d].mean()), 3),
            }
            for d in sorted(set(d_all))
        },
    )
