"""Candidate (hypothetical) parameter values and model inference for them. Nothing is hard-coded."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.counterfactual.constraints import intervention_scale
from src.features.feature_engineering import indicator_name


def candidate_values(v: float, c: pd.Series, cf: dict) -> np.ndarray:
    """Grid anchored at the current value, step = grid_step_pct*scale, limited by max change and the operating range."""
    scale = intervention_scale(v, c["span"], cf["scale_span_floor"])
    lim, step, min_d = cf["max_change_pct"] * scale, cf["grid_step_pct"] * scale, cf["min_change_pct"] * scale
    lo = max(c["op_lo"], c["obs_min"], v - lim)
    hi = min(c["op_hi"], c["obs_max"], v + lim)
    if lo > hi:
        return np.array([])
    K = int(np.ceil(cf["max_change_pct"] / cf["grid_step_pct"]))
    vals = v + step * np.arange(-K, K + 1)
    vals = vals[(vals >= lo - 1e-12) & (vals <= hi + 1e-12)]
    pm = float(c["pass_median"])                       # also try "move to the PASS median" when reachable
    if lo <= pm <= hi:
        vals = np.append(vals, pm)
    vals = vals[np.abs(vals - v) >= min_d]
    return np.unique(np.round(vals, 10))


def predict_with_values(model, x_row: pd.Series, model_features: list[str], changes: dict[str, np.ndarray]) -> np.ndarray:
    """Predict FAIL probability for hypothetical rows. `changes` maps feature -> equal-length value arrays."""
    n = len(next(iter(changes.values())))
    base = x_row[model_features].to_numpy(float)
    M = np.tile(base, (n, 1))
    for f, vals in changes.items():
        M[:, model_features.index(f)] = vals
        ind = indicator_name(f)
        if ind in model_features:                      # a set value is, by definition, an observed value
            M[:, model_features.index(ind)] = 0.0
    return model.predict_proba(pd.DataFrame(M, columns=model_features))[:, 1]
