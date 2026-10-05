"""Safe-intervention constraints learned from the training distribution."""
from __future__ import annotations

import numpy as np
import pandas as pd


def build_constraints(X_raw: pd.DataFrame, y, sensor_cols: list[str], cf_cfg: dict) -> pd.DataFrame:
    lo_p, hi_p = cf_cfg["operating_percentiles"]
    A = X_raw[sensor_cols].to_numpy(float)
    pm = np.asarray(y).astype(int) == 0
    t = pd.DataFrame(index=sensor_cols)
    t["op_lo"], t["op_hi"] = np.nanpercentile(A, lo_p, axis=0), np.nanpercentile(A, hi_p, axis=0)
    p5, p95 = np.nanpercentile(A, 5, axis=0), np.nanpercentile(A, 95, axis=0)
    t["span"] = p95 - p5
    t["obs_min"], t["obs_max"] = np.nanmin(A, axis=0), np.nanmax(A, axis=0)
    P = A[pm]
    t["pass_p5"], t["pass_p95"] = np.nanpercentile(P, 5, axis=0), np.nanpercentile(P, 95, axis=0)
    t["pass_median"] = np.nanmedian(P, axis=0)
    return t


def intervention_scale(value: float, span: float, floor_frac: float = 0.25) -> float:
    """Denominator for percentage limits: max(|value|, floor_frac * P5-P95 span).

    Plain "% of value" breaks for values near zero or negative; the span floor keeps limits meaningful there
    without inflating them for heavy-tailed sensors (a full-span floor would).
    """
    return max(abs(float(value)), floor_frac * float(span), 1e-9)
