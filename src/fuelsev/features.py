"""Window features for driving-behaviour anomaly detection.

The same function builds features offline (for training and evaluation, over a recorded
session) and online (inside the in-cab agent, over the last few seconds), so the model
sees identical inputs in both places. Features use only telemetry fields, never the
simulator's hidden state.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from .schemas import Telemetry

WINDOW_S = 10
HOP_S = 5
MIN_MOVING_KMH = 5.0  # windows slower than this on average are not "driving"

FEATURE_NAMES = (
    "speed_mean",
    "speed_max",
    "speed_std",
    "speed_range",
    "accel_min",
    "accel_max",
    "jerk_rms",
    "accel_flips",
    "rpm_mean",
    "rpm_max",
    "throttle_mean",
    "throttle_std",
    "brake_frac",
    "limit_ratio_mean",
    "limit_ratio_max",
    "neutral_frac",
    "grade_mean",
    "payload_t",
)
N_FEATURES = len(FEATURE_NAMES)


def window_features(window: Sequence[Telemetry]) -> np.ndarray:
    """Feature vector for one window of consecutive 1 Hz samples."""
    speed = np.fromiter((t.speed_kmh for t in window), float)
    accel = np.fromiter((t.accel_ms2 for t in window), float)
    rpm = np.fromiter((t.rpm for t in window), float)
    thr = np.fromiter((t.throttle_pct for t in window), float)
    brk = np.fromiter((t.brake_pct for t in window), float)
    gear = np.fromiter((t.gear for t in window), float)
    limit = np.fromiter((t.speed_limit_kmh for t in window), float)
    grade = np.fromiter((t.grade_pct for t in window), float)
    payload = np.fromiter((t.payload_t for t in window), float)

    jerk = np.diff(accel)
    strong = accel[np.abs(accel) > 0.25]
    flips = int(np.sum(np.diff(np.sign(strong)) != 0)) if strong.size > 1 else 0
    ratio = speed / np.maximum(limit, 1.0)

    return np.array(
        [
            speed.mean(),
            speed.max(),
            speed.std(),
            speed.max() - speed.min(),
            accel.min(),
            accel.max(),
            float(np.sqrt(np.mean(jerk**2))) if jerk.size else 0.0,
            flips,
            rpm.mean(),
            rpm.max(),
            thr.mean(),
            thr.std(),
            float(np.mean(brk > 5.0)),
            ratio.mean(),
            ratio.max(),
            float(np.mean((gear == 0) & (speed > 10.0))),
            grade.mean(),
            payload.mean(),
        ],
        dtype=float,
    )


def is_driving(window: Sequence[Telemetry]) -> bool:
    return float(np.mean([t.speed_kmh for t in window])) >= MIN_MOVING_KMH


def session_windows(
    telemetry: Sequence[Telemetry],
    unsafe: Sequence[bool] | None = None,
    window_s: int = WINDOW_S,
    hop_s: int = HOP_S,
) -> tuple[np.ndarray, np.ndarray]:
    """Slide a window over a session and return (X, y).

    ``y`` is 1 where at least half the window is labelled unsafe. If ``unsafe`` is not
    given, ``y`` is all zeros. Windows when the truck is not driving are skipped.
    """
    xs: list[np.ndarray] = []
    ys: list[int] = []
    for start in range(0, len(telemetry) - window_s + 1, hop_s):
        w = telemetry[start : start + window_s]
        if not is_driving(w):
            continue
        xs.append(window_features(w))
        if unsafe is not None:
            ys.append(int(np.mean(unsafe[start : start + window_s]) >= 0.5))
        else:
            ys.append(0)
    if not xs:
        return np.empty((0, N_FEATURES)), np.empty((0,), dtype=int)
    return np.vstack(xs), np.asarray(ys, dtype=int)
