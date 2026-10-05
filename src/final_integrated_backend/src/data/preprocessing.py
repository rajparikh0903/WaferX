"""Leak-free, reusable preprocessing: cleaning, median imputation, optional missing indicators, scaling.

Everything learned (dropped columns, medians, scaler) is fitted on the TRAINING split only.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.data.validation import dominant_fraction
from src.features.feature_engineering import indicator_name, missing_indicator_frame, is_indicator


def make_splits(y: pd.Series, ts: pd.Series, cfg: dict) -> dict[str, np.ndarray]:
    """Return index arrays (sample ids) for train / val / test."""
    sc, seed, n = cfg["split"], cfg["project"]["seed"], len(y)
    n_tr, n_va = int(n * sc["train_frac"]), int(n * sc["val_frac"])
    if sc["method"] == "temporal":
        order = np.argsort(ts.fillna(ts.max()).to_numpy(), kind="stable")
        parts = {"train": order[:n_tr], "val": order[n_tr:n_tr + n_va], "test": order[n_tr + n_va:]}
    elif sc["method"] == "stratified":
        idx = np.arange(n)
        tr, rest = train_test_split(idx, train_size=n_tr, stratify=y, random_state=seed)
        va, te = train_test_split(rest, train_size=n_va, stratify=y.iloc[rest], random_state=seed)
        parts = {"train": tr, "val": va, "test": te}
    else:
        raise ValueError(f"unknown split method {sc['method']}")
    # Preserve the order produced by train_test_split for stratified experiments.
    # LightGBM's row-order-sensitive subsampling can otherwise change the fitted model.
    if sc["method"] == "stratified":
        return parts
    return {k: np.sort(v) for k, v in parts.items()}


class SecomPreprocessor:
    def __init__(self, max_missing_frac=0.5, max_dominant_frac=0.995, indicator_min_missing_frac=0.05,
                 use_indicators=False, version="1.0.0"):
        self.max_missing_frac = max_missing_frac
        self.max_dominant_frac = max_dominant_frac
        self.indicator_min_missing_frac = indicator_min_missing_frac
        self.use_indicators = use_indicators
        self.version = version
        self.fitted = False

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _coerce(X: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(X, pd.DataFrame):
            raise ValueError("X must be a pandas DataFrame")
        Xn = X.apply(pd.to_numeric, errors="coerce")
        if bool((Xn.isna() & X.notna()).to_numpy().any()):
            raise ValueError("Non-numeric values found in feature matrix")
        if bool(np.isinf(Xn.to_numpy(dtype=float)).any()):
            raise ValueError("Infinite values found in feature matrix")
        return Xn.astype(float)

    def _check_columns(self, X: pd.DataFrame) -> None:
        unknown = [c for c in X.columns if c not in set(self.input_columns_)]
        missing = [c for c in self.kept_columns_ if c not in X.columns]
        if unknown or missing:
            raise ValueError(f"Feature mismatch: {len(unknown)} unknown columns (e.g. {unknown[:3]}), "
                             f"{len(missing)} required columns missing (e.g. {missing[:3]})")

    # ------------------------------------------------------------------ fit / transform
    def fit(self, X: pd.DataFrame) -> "SecomPreprocessor":
        X = self._coerce(X)
        self.input_columns_ = list(X.columns)
        miss = X.isna().mean()
        nun = X.nunique(dropna=True)
        dom = X.apply(dominant_fraction)
        drop_missing = miss[miss > self.max_missing_frac].index.tolist()
        drop_const = nun[nun <= 1].index.tolist()
        drop_nzv = [c for c in dom[dom >= self.max_dominant_frac].index if c not in drop_const]
        dropped = set(drop_missing) | set(drop_const) | set(drop_nzv)
        self.kept_columns_ = [c for c in X.columns if c not in dropped]
        self.medians_ = X[self.kept_columns_].median()
        self.missing_frac_ = miss[self.kept_columns_]
        self.indicator_columns_ = [c for c in self.kept_columns_
                                   if miss[c] >= self.indicator_min_missing_frac]
        self.scaler_ = StandardScaler().fit(X[self.kept_columns_].fillna(self.medians_))
        self.report_ = {
            "n_input_columns": len(self.input_columns_),
            "n_kept_sensor_columns": len(self.kept_columns_),
            "dropped_missing_gt_threshold": drop_missing,
            "dropped_constant": drop_const,
            "dropped_near_constant": drop_nzv,
            "n_indicator_candidates": len(self.indicator_columns_),
            "imputation": "median (train split)",
        }
        self.fitted = True
        return self

    @property
    def model_columns(self) -> list[str]:
        cols = list(self.kept_columns_)
        if self.use_indicators:
            cols += [indicator_name(c) for c in self.indicator_columns_]
        return cols

    @property
    def sensor_columns(self) -> list[str]:
        return list(self.kept_columns_)

    def transform(self, X: pd.DataFrame, with_indicators: bool | None = None) -> pd.DataFrame:
        """Imputed raw-unit matrix (+ optional missing indicators). Trees need no scaling."""
        assert self.fitted, "call fit first"
        X = self._coerce(X)
        self._check_columns(X)
        raw = X.reindex(columns=self.kept_columns_)
        out = raw.fillna(self.medians_)
        use_ind = self.use_indicators if with_indicators is None else with_indicators
        if use_ind and self.indicator_columns_:
            out = pd.concat([out, missing_indicator_frame(raw, self.indicator_columns_)], axis=1)
        return out

    def transform_scaled(self, X: pd.DataFrame) -> pd.DataFrame:
        base = self.transform(X, with_indicators=False)
        return pd.DataFrame(self.scaler_.transform(base), index=base.index, columns=base.columns)

    def raw_sensors(self, X: pd.DataFrame) -> pd.DataFrame:
        """Kept sensors with NaN preserved (used for honest PASS/FAIL distribution statistics)."""
        return self._coerce(X).reindex(columns=self.kept_columns_)
