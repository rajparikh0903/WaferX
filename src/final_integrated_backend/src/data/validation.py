"""Data-quality report for SECOM (shape, duplicates, missingness, constants, target, timestamps, drift)."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from src.utils import resolve, to_py


def dominant_fraction(s: pd.Series) -> float:
    s = s.dropna()
    if s.empty:
        return 1.0
    return float(s.value_counts(normalize=True).iloc[0])


def data_quality_report(X: pd.DataFrame, y: pd.Series, ts: pd.Series, cfg: dict) -> dict:
    n, p = X.shape
    Xn = X.apply(pd.to_numeric, errors="coerce")
    arr = Xn.to_numpy(dtype=float)
    miss = Xn.isna().mean()
    nun = Xn.nunique(dropna=True)
    const = nun[nun <= 1].index.tolist()
    dom = Xn.apply(dominant_fraction)
    thr = cfg["preprocessing"]["max_dominant_frac"]
    nzv = [c for c in dom[dom >= thr].index if c not in const]
    n_fail = int(y.sum())
    order = np.argsort(ts.fillna(ts.max()).to_numpy(), kind="stable")
    chunks = np.array_split(order, 5)
    drift = [{
        "chunk": i + 1,
        "from": ts.iloc[c].min(), "to": ts.iloc[c].max(),
        "n": int(len(c)), "fail_rate": float(y.iloc[c].mean()),
    } for i, c in enumerate(chunks)]
    rep = {
        "n_rows": int(n), "n_columns": int(p),
        "n_duplicate_rows": int(X.duplicated().sum()),
        "non_numeric_cells": int((Xn.isna() & X.notna()).sum().sum()),
        "infinite_cells": int(np.isinf(arr).sum()),
        "missing": {
            "total_fraction": float(np.isnan(arr).mean()),
            "columns_with_any_missing": int((miss > 0).sum()),
            "columns_missing_over_50pct": int((miss > 0.5).sum()),
            "columns_missing_over_max_frac": int((miss > cfg["preprocessing"]["max_missing_frac"]).sum()),
            "worst_10_columns": miss.sort_values(ascending=False).head(10).round(4).to_dict(),
            "rows_with_any_missing": int(Xn.isna().any(axis=1).sum()),
        },
        "constant_features": {"count": len(const), "names": const},
        "near_zero_variance_features": {"rule": f"dominant value share >= {thr}", "count": len(nzv), "names": nzv},
        "target": {
            "n_pass": int(n - n_fail), "n_fail": n_fail, "fail_rate": float(n_fail / n),
            "imbalance_ratio_pass_to_fail": float((n - n_fail) / max(n_fail, 1)),
        },
        "timestamps": {
            "invalid_count": int(ts.isna().sum()),
            "min": ts.min(), "max": ts.max(),
            "monotonic_increasing_in_file_order": bool(ts.is_monotonic_increasing),
            "duplicate_timestamps": int(ts.duplicated().sum()),
        },
        "temporal_drift_fail_rate_by_time_chunk": drift,
    }
    f = [d["fail_rate"] for d in drift]
    rep["split_recommendation"] = (
        "temporal: samples are time-ordered, production data arrives in time order, and "
        f"fail rate varies across time chunks ({min(f):.3f}-{max(f):.3f}), so a random split would leak "
        "future process behaviour into training and give optimistic metrics."
    )
    return to_py(rep)


def write_quality_report(rep: dict, cfg: dict) -> None:
    out = resolve(cfg, "processed_dir")
    (out / "data_quality_report.json").write_text(json.dumps(rep, indent=2))
    t, m = rep["target"], rep["missing"]
    lines = [
        "# SECOM data-quality report", "",
        f"- Rows x columns: **{rep['n_rows']} x {rep['n_columns']}**",
        f"- Duplicate rows: {rep['n_duplicate_rows']}; non-numeric cells: {rep['non_numeric_cells']}; infinite cells: {rep['infinite_cells']}",
        f"- Missing cells: {m['total_fraction']:.2%}; columns with any missing: {m['columns_with_any_missing']}; "
        f"columns > 50% missing: {m['columns_missing_over_50pct']}; rows with any missing: {m['rows_with_any_missing']}",
        f"- Constant features: {rep['constant_features']['count']}; near-zero-variance: {rep['near_zero_variance_features']['count']}",
        f"- Target: PASS {t['n_pass']} / FAIL {t['n_fail']} (fail rate {t['fail_rate']:.2%}, imbalance {t['imbalance_ratio_pass_to_fail']:.1f}:1)",
        f"- Timestamps: {rep['timestamps']['min']} -> {rep['timestamps']['max']}; invalid {rep['timestamps']['invalid_count']}; "
        f"monotonic {rep['timestamps']['monotonic_increasing_in_file_order']}",
        "", "## Fail rate by time chunk", "", "| chunk | from | to | n | fail rate |", "|---|---|---|---|---|",
    ]
    for d in rep["temporal_drift_fail_rate_by_time_chunk"]:
        lines.append(f"| {d['chunk']} | {d['from']} | {d['to']} | {d['n']} | {d['fail_rate']:.3f} |")
    lines += ["", "## Split decision", "", rep["split_recommendation"]]
    (out / "data_quality_report.md").write_text("\n".join(lines))
