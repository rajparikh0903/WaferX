"""Evidence primitives for root-cause candidates (all statistics come from the TRAINING split).

Evidence types
  1 SHAP contribution      (computed by the ShapEngine, normalised in candidate_engine)
  2 Feature deviation      robust z-score vs PASS population (median / 1.4826*MAD, std fallback)
  3 Failure association    PASS-vs-FAIL Mann-Whitney U, rank-biserial effect size, FDR-adjusted p-values
  4 Historical association similar historical FAIL samples that deviate in the same feature/direction
  5 External anomaly       accepted from another model via analyze_anomaly(); never computed here
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def analyze_anomaly(sample_id=None, anomaly_score=None, anomaly_label=None) -> dict:
    """Interface for the externally developed anomaly model. Everything is optional.

    anomaly_score must be normalised to [0, 1]; anomaly_label must be a bool. If only the label is given it
    is used as a 1.0 / 0.0 score. The root-cause engine works unchanged when no anomaly info is available.
    """
    score = None
    if anomaly_score is not None:
        if isinstance(anomaly_score, bool):
            raise ValueError("anomaly_score must be a number in [0, 1]")
        try:
            score = float(anomaly_score)
        except (TypeError, ValueError):
            raise ValueError("anomaly_score must be a number in [0, 1]")
        if not np.isfinite(score) or not 0.0 <= score <= 1.0:
            raise ValueError("anomaly_score must be within [0, 1]")
    if anomaly_label is not None and not isinstance(anomaly_label, (bool, np.bool_)):
        raise ValueError("anomaly_label must be a boolean")
    label = None if anomaly_label is None else bool(anomaly_label)
    effective = score if score is not None else (None if label is None else (1.0 if label else 0.0))
    return {"available": effective is not None, "sample_id": sample_id, "anomaly_score": score,
            "anomaly_label": label, "effective_score": effective,
            "source": "external anomaly model" if effective is not None else None}


def _block(v: np.ndarray) -> dict | None:
    if len(v) == 0:
        return None
    return {"n": int(len(v)), "mean": float(v.mean()), "median": float(np.median(v)),
            "std": float(v.std(ddof=1)) if len(v) > 1 else 0.0,
            "p5": float(np.percentile(v, 5)), "p95": float(np.percentile(v, 95))}


def _pct(sorted_vals: np.ndarray, v: float) -> float | None:
    if len(sorted_vals) == 0:
        return None
    return float(100.0 * np.searchsorted(sorted_vals, v, side="right") / len(sorted_vals))


class ReferenceStats:
    """PASS/FAIL reference statistics learned from the training split."""

    def __init__(self, X_model: pd.DataFrame, X_raw: pd.DataFrame, y, sensor_cols: list[str], rc_cfg: dict):
        self.sensor_cols = list(sensor_cols)
        self.index = X_model.index.to_numpy()
        self.y = np.asarray(y).astype(int)
        self.n_pass, self.n_fail = int((self.y == 0).sum()), int((self.y == 1).sum())
        raw = X_raw[self.sensor_cols].to_numpy(float)
        pm = self.y == 0
        self.pass_obs, self.fail_obs = {}, {}
        med, scale, std, mean = [], [], [], []
        for j, c in enumerate(self.sensor_cols):
            pv, fv = raw[pm, j], raw[~pm, j]
            pv, fv = np.sort(pv[~np.isnan(pv)]), np.sort(fv[~np.isnan(fv)])
            self.pass_obs[c], self.fail_obs[c] = pv, fv
            m = float(np.median(pv)) if len(pv) else 0.0
            sd = float(pv.std(ddof=1)) if len(pv) > 1 else 0.0
            mad = 1.4826 * float(np.median(np.abs(pv - m))) if len(pv) else 0.0
            s = mad if mad > 1e-12 else (sd if sd > 1e-12 else 1.0)
            med.append(m); scale.append(s); std.append(sd); mean.append(float(pv.mean()) if len(pv) else 0.0)
        self.pass_median = np.array(med); self.scale = np.array(scale)
        self.pass_std = np.array(std); self.pass_mean = np.array(mean)
        self.Z = ((X_model[self.sensor_cols].to_numpy(float) - self.pass_median) / self.scale).astype(np.float32)
        self.assoc = self._association(rc_cfg)

    def _association(self, rc_cfg) -> pd.DataFrame:
        rows = []
        for c in self.sensor_cols:
            pv, fv = self.pass_obs[c], self.fail_obs[c]
            if len(pv) >= 5 and len(fv) >= 5:
                u = stats.mannwhitneyu(fv, pv, alternative="two-sided")
                auc = u.statistic / (len(fv) * len(pv))
                rows.append((c, auc, 2 * auc - 1, float(u.pvalue)))
            else:
                rows.append((c, 0.5, 0.0, 1.0))
        df = pd.DataFrame(rows, columns=["feature", "auc", "effect_r", "p_value"]).set_index("feature")
        df["q_value"] = stats.false_discovery_control(df["p_value"].to_numpy(), method="bh")
        df["effect_pct"] = stats.rankdata(df["effect_r"].abs()) / len(df)
        df["significant"] = df["q_value"] < rc_cfg.get("fdr_alpha", 0.05)
        return df

    def robust_z(self, x_row: pd.Series) -> np.ndarray:
        return (x_row[self.sensor_cols].to_numpy(float) - self.pass_median) / self.scale

    def compare(self, feature: str, value: float, bins: int = 20) -> dict:
        j = self.sensor_cols.index(feature)
        pv, fv = self.pass_obs[feature], self.fail_obs[feature]
        z = (value - self.pass_median[j]) / self.scale[j]
        pb, fb = _block(pv), _block(fv)
        closer = None
        if fb is not None and pb is not None:
            closer = "FAIL" if abs(value - fb["median"]) < abs(value - pb["median"]) else "PASS"
        both = np.concatenate([pv, fv])
        hist = None
        if len(both) > 5:
            lo, hi = np.percentile(both, [1, 99])
            hi = hi if hi > lo else lo + 1.0
            edges = np.linspace(lo, hi, bins + 1)
            dens = lambda a: (np.histogram(np.clip(a, lo, hi), bins=edges)[0] / max(len(a), 1)).tolist()
            hist = {"edges": edges.tolist(), "pass_fraction": dens(pv), "fail_fraction": dens(fv)}
        return {"feature": feature, "current_value": float(value), "pass": pb, "fail": fb,
                "percentile_in_pass": _pct(pv, value), "percentile_in_fail": _pct(fv, value),
                "deviation_sigma": float(z),
                "deviation_sigma_mean_std": float((value - self.pass_mean[j]) / self.pass_std[j]) if self.pass_std[j] > 0 else None,
                "closer_to": closer, "histogram": hist}
