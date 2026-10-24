"""Isolation Forest anomaly detection on driving-window features.

The model is unsupervised. It learns what normal windows look like from a baseline fleet
(speed, braking, revs, throttle smoothness, speed relative to the limit, neutral use) and
scores how isolated a new window is. A window is flagged when its score falls below a
threshold set at a chosen false-alarm rate on the training data.

This catches behaviour no single fixed rule describes, such as erratic speed control from
fatigue or an unusual mix of high speed, hard braking and revs, which complements the
rules in ``rules.py``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import sklearn
from sklearn.ensemble import IsolationForest
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler

from .features import FEATURE_NAMES, N_FEATURES


@dataclass
class AnomalyModel:
    pipeline: Pipeline
    threshold: float
    centre: np.ndarray  # training median per feature
    spread: np.ndarray  # training inter-quartile range per feature
    sklearn_version: str = sklearn.__version__

    def scores(self, X: np.ndarray) -> np.ndarray:
        """Higher means more normal; lower means more anomalous."""
        X = np.atleast_2d(X)
        return self.pipeline.decision_function(X)

    def is_anomalous(self, X: np.ndarray) -> np.ndarray:
        return self.scores(X) < self.threshold

    def explain(self, x: np.ndarray, k: int = 3) -> list[tuple[str, float]]:
        """The k features furthest from normal, as robust z-scores (signed)."""
        z = (np.asarray(x, dtype=float) - self.centre) / self.spread
        order = np.argsort(-np.abs(z))[:k]
        return [(FEATURE_NAMES[i], float(round(z[i], 2))) for i in order]

    # -- persistence ---------------------------------------------------------------
    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: str | Path) -> AnomalyModel:
        """Load a model saved by ``save``. Only load files you created yourself.

        Joblib uses pickle, which can execute code, so never load a model from an untrusted source.
        """
        model = joblib.load(path)
        if not isinstance(model, cls):
            raise TypeError(f"{path} does not contain an AnomalyModel")
        if model.sklearn_version.split(".")[:2] != sklearn.__version__.split(".")[:2]:
            raise RuntimeError(
                f"model was trained with scikit-learn {model.sklearn_version} but {sklearn.__version__} "
                "is installed; retrain with `fuelsev train`"
            )
        return model


def train_anomaly_model(
    X: np.ndarray,
    false_alarm_rate: float = 0.02,
    n_estimators: int = 200,
    seed: int = 0,
) -> AnomalyModel:
    """Fit on baseline windows and set the alert threshold.

    ``false_alarm_rate`` is the fraction of the training windows that will score as anomalous.
    """
    X = np.asarray(X, dtype=float)
    if X.ndim != 2 or X.shape[1] != N_FEATURES:
        raise ValueError(f"expected an (n, {N_FEATURES}) feature matrix, got {X.shape}")
    if X.shape[0] < 50:
        raise ValueError("need at least 50 training windows")
    if not 0.0 < false_alarm_rate < 0.5:
        raise ValueError("false_alarm_rate must be between 0 and 0.5")

    pipe = Pipeline(
        [
            ("scale", RobustScaler()),
            ("forest", IsolationForest(n_estimators=n_estimators, random_state=seed, n_jobs=1)),
        ]
    )
    pipe.fit(X)
    scores = pipe.decision_function(X)
    threshold = float(np.quantile(scores, false_alarm_rate))

    q1, med, q3 = np.percentile(X, [25, 50, 75], axis=0)
    spread = np.where((q3 - q1) > 1e-6, q3 - q1, np.maximum(X.std(axis=0), 1e-6))
    return AnomalyModel(pipe, threshold, med, spread)
