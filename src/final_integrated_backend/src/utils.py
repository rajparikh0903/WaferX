"""Shared helpers: config loading, path resolution, JSON-safe conversion."""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]

DISCLAIMER = (
    "This is a model-based counterfactual recommendation, not a physically validated "
    "process correction. Real semiconductor process changes require engineering "
    "validation and safety constraints."
)
CAUSALITY_NOTE = (
    "Root-cause candidates are hypotheses ranked by model and statistical evidence; "
    "SECOM has no ground-truth causal labels, so none of them is a proven root cause."
)


def load_config(path: str | Path | None = None) -> dict:
    p = Path(path) if path else ROOT / "config.yaml"
    with open(p, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["_root"] = str(p.resolve().parent)
    return cfg


def resolve(cfg: dict, key: str) -> Path:
    p = Path(cfg["_root"]) / cfg["paths"][key]
    p.mkdir(parents=True, exist_ok=True)
    return p


def to_py(obj):
    """Recursively convert numpy / pandas objects to JSON-safe python (NaN/inf -> None)."""
    if isinstance(obj, dict):
        return {str(k): to_py(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [to_py(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return [to_py(v) for v in obj.tolist()]
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        f = float(obj)
        return None if (math.isnan(f) or math.isinf(f)) else f
    if isinstance(obj, pd.Timestamp):
        return None if pd.isna(obj) else obj.isoformat()
    if obj is pd.NaT:
        return None
    if isinstance(obj, pd.Series):
        return to_py(obj.to_dict())
    if isinstance(obj, pd.DataFrame):
        return to_py(obj.to_dict(orient="records"))
    return obj
