"""SECOM anomaly detector.

The detector is an actual unsupervised Isolation Forest trained only on the
TRAIN split PASS rows.  It is intentionally separate from the supervised
PASS/FAIL models.

Score semantics:
  - 0.0 .. 1.0 empirical percentile of the Isolation Forest anomaly score
  - higher = more unusual relative to the PASS training population
  - anomaly_label is true when score >= the calibrated percentile threshold

Important: SECOM does not provide ground-truth anomaly labels.  FAIL is not
used as the anomaly target; it is only a proxy for evaluation reporting.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd


MODEL_PATH = (
    Path(__file__).resolve().parents[2]
    / "models"
    / "anomaly"
    / "isolation_forest_pass.joblib"
)


class SECOMAnomalyDetector:
    def __init__(self, model_path: Path = MODEL_PATH):
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"Anomaly model not found: {self.model_path}")

        artifact: dict[str, Any] = joblib.load(self.model_path)
        required = {
            "model",
            "scaler",
            "feature_list",
            "reference_scores_sorted",
            "label_threshold",
        }
        missing = sorted(required - set(artifact))
        if missing:
            raise RuntimeError(
                f"Anomaly artifact is missing required keys: {missing}"
            )

        self.model = artifact["model"]
        self.scaler = artifact["scaler"]
        self.feature_list = list(artifact["feature_list"])
        self.reference_scores_sorted = np.asarray(
            artifact["reference_scores_sorted"], dtype=float
        )
        self.label_threshold = float(artifact["label_threshold"])
        self.score_method = str(artifact.get("score_method", "unknown"))
        self.train_pass_count = int(artifact.get("train_pass_count", 0))
        self.train_split_size = int(artifact.get("train_split_size", 0))
        self.train_on = str(artifact.get("train_on", "unknown"))
        self.evaluation_proxy = str(artifact.get("evaluation_proxy", ""))

        if self.reference_scores_sorted.size == 0:
            raise RuntimeError("Anomaly reference score distribution is empty")

    def _matrix(self, X: pd.DataFrame | np.ndarray) -> pd.DataFrame:
        if isinstance(X, pd.DataFrame):
            if set(X.columns) != set(self.feature_list):
                raise ValueError(
                    "Anomaly feature mismatch. Expected the saved model feature list."
                )
            frame = X.reindex(columns=self.feature_list).copy()
        else:
            arr = np.asarray(X, dtype=float)
            if arr.ndim == 1:
                arr = arr.reshape(1, -1)
            frame = pd.DataFrame(arr, columns=self.feature_list)

        if frame.ndim != 2 or frame.shape[1] != len(self.feature_list):
            raise ValueError(
                f"Anomaly model expects {len(self.feature_list)} model features; "
                f"received shape {frame.shape}."
            )
        if not np.isfinite(frame.to_numpy(dtype=float)).all():
            raise ValueError("Anomaly inference received non-finite feature values")
        return frame

    def score(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        frame = self._matrix(X)
        scaled = self.scaler.transform(frame)
        raw = -self.model.decision_function(scaled)
        # Empirical percentile in [0,1] against the PASS training reference.
        score = np.searchsorted(
            self.reference_scores_sorted,
            raw,
            side="right",
        ) / len(self.reference_scores_sorted)
        return np.clip(score, 0.0, 1.0)

    def analyze(self, X: pd.DataFrame | np.ndarray, sample_id: int | None = None) -> dict:
        score = float(self.score(X)[0])
        label = bool(score >= self.label_threshold)

        if score >= 0.97:
            severity = "CRITICAL"
        elif score >= self.label_threshold:
            severity = "HIGH"
        elif score >= 0.75:
            severity = "MEDIUM"
        else:
            severity = "LOW"

        return {
            "available": True,
            "sample_id": sample_id,
            "anomaly_score": score,
            "anomaly_label": label,
            "effective_score": score,
            "source": "Isolation Forest (train-PASS calibrated)",
            "model": "IsolationForest",
            "model_type": "unsupervised_anomaly_detection",
            "alert_threshold": self.label_threshold,
            "severity": severity,
            "score_semantics": "empirical percentile of anomaly score relative to TRAIN PASS population",
            "train_pass_count": self.train_pass_count,
            "train_split_size": self.train_split_size,
            "evaluation_note": self.evaluation_proxy,
        }

    def info(self) -> dict:
        return {
            "model": "IsolationForest",
            "model_type": "unsupervised_anomaly_detection",
            "artifact": str(self.model_path),
            "n_features": len(self.feature_list),
            "n_estimators": getattr(self.model, "n_estimators", None),
            "max_samples": getattr(self.model, "max_samples", None),
            "max_features": getattr(self.model, "max_features", None),
            "random_state": getattr(self.model, "random_state", None),
            "label_threshold": self.label_threshold,
            "score_method": self.score_method,
            "train_pass_count": self.train_pass_count,
            "train_split_size": self.train_split_size,
            "train_on": self.train_on,
            "evaluation_proxy": self.evaluation_proxy,
        }


anomaly_detector = SECOMAnomalyDetector()
