"""Train the SECOM Isolation Forest anomaly detector.

Leakage-safe design:
  - load official SECOM data
  - use the existing train-only preprocessor
  - fit the anomaly scaler + Isolation Forest on TRAIN split PASS rows only
  - calibrate the returned 0..1 anomaly score as an empirical percentile
    against the TRAIN PASS anomaly-score distribution

SECOM does not provide anomaly labels, so FAIL is used only as an evaluation
proxy in the report; it is never the anomaly training target.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

from src.data.loader import load_secom
from src.data.preprocessing import make_splits
from src.utils import load_config, resolve


def percentile_score(raw: np.ndarray, reference_sorted: np.ndarray) -> np.ndarray:
    return np.clip(
        np.searchsorted(reference_sorted, raw, side="right")
        / len(reference_sorted),
        0.0,
        1.0,
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-estimators", type=int, default=200)
    ap.add_argument("--max-samples", type=float, default=0.70)
    ap.add_argument("--max-features", type=float, default=1.0)
    ap.add_argument("--threshold", type=float, default=0.90)
    ap.add_argument("--random-state", type=int, default=42)
    args = ap.parse_args()

    cfg = load_config()
    root = Path(cfg["_root"])
    pre = joblib.load(resolve(cfg, "preprocessing_dir") / "preprocessor.joblib")
    X, y, ts = load_secom(cfg)
    splits = make_splits(y, ts, cfg)

    # Existing preprocessor was fit on TRAIN only during the supervised model build.
    # It returns imputed model-feature values without labels.
    X_model = pre.transform(X)

    train_idx = splits["train"]
    val_idx = splits["val"]
    test_idx = splits["test"]
    train_pass = train_idx[y.iloc[train_idx].to_numpy() == 0]

    scaler = StandardScaler().fit(X_model.iloc[train_pass])
    X_train_pass = scaler.transform(X_model.iloc[train_pass])

    model = IsolationForest(
        n_estimators=args.n_estimators,
        max_samples=args.max_samples,
        max_features=args.max_features,
        contamination="auto",
        random_state=args.random_state,
        n_jobs=-1,
    )
    model.fit(X_train_pass)

    reference_sorted = np.sort(-model.decision_function(X_train_pass))

    def score_indices(indices):
        raw = -model.decision_function(scaler.transform(X_model.iloc[indices]))
        return percentile_score(raw, reference_sorted)

    metrics = {
        "train": {},
        "validation": {},
        "test": {},
        "note": "FAIL is only an evaluation proxy; SECOM has no ground-truth anomaly labels.",
    }

    for name, indices in [
        ("train", train_idx),
        ("validation", val_idx),
        ("test", test_idx),
    ]:
        score = score_indices(indices)
        yy = y.iloc[indices].to_numpy()
        metrics[name] = {
            "roc_auc_fail_proxy": float(roc_auc_score(yy, score)),
            "pr_auc_fail_proxy": float(average_precision_score(yy, score)),
            "anomaly_rate_at_threshold": float(np.mean(score >= args.threshold)),
            "pass_anomaly_rate": float(np.mean(score[yy == 0] >= args.threshold)),
            "fail_anomaly_rate": float(np.mean(score[yy == 1] >= args.threshold)),
        }

    out_dir = root / "models" / "anomaly"
    out_dir.mkdir(parents=True, exist_ok=True)

    artifact = {
        "model": model,
        "scaler": scaler,
        "feature_list": list(X_model.columns),
        "reference_scores_sorted": reference_sorted,
        "label_threshold": float(args.threshold),
        "score_method": "empirical_percentile_of_negative_isolation_forest_decision_score",
        "train_pass_count": int(len(train_pass)),
        "train_split_size": int(len(train_idx)),
        "random_state": int(args.random_state),
        "train_on": "training split PASS samples only",
        "evaluation_proxy": "SECOM FAIL label used only as a proxy for anomaly evaluation; no ground-truth anomaly labels",
    }
    joblib.dump(artifact, out_dir / "isolation_forest_pass.joblib", compress=3)
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print(json.dumps({"artifact": str(out_dir / "isolation_forest_pass.joblib"), "metrics": metrics}, indent=2))


if __name__ == "__main__":
    main()
