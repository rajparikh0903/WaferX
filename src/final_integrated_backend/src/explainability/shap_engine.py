"""Tree-model explainability for XGBoost and LightGBM."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.features.feature_engineering import is_indicator


class ShapEngine:
    def __init__(self, model, feature_names: list[str]):
        import shap
        self.model = model
        self.feature_names = list(feature_names)
        self.model_type = type(model).__name__
        self.explainer = None
        self.base_value = None
        try:
            self.explainer = shap.TreeExplainer(model)
            ev = self.explainer.expected_value
            self.base_value = float(np.ravel(ev)[-1])
        except Exception:
            self.explainer = None
        if self.model_type.startswith("XGB"):
            # XGBoost's native pred_contribs bias is the exact additive base in model margin.
            self.base_value = self._native_base_value()
        elif self.base_value is None:
            self.base_value = self._native_base_value()

    def _frame(self, X) -> pd.DataFrame:
        if isinstance(X, pd.DataFrame):
            return X[self.feature_names]
        return pd.DataFrame(np.asarray(X), columns=self.feature_names)

    def _native_contrib(self, X: pd.DataFrame) -> np.ndarray:
        if hasattr(self.model, "booster_"):
            # LightGBM: last column is the bias / expected value.
            return np.asarray(self.model.booster_.predict(X, pred_contrib=True))
        if hasattr(self.model, "get_booster"):
            import xgboost as xgb
            dm = xgb.DMatrix(X, feature_names=self.feature_names)
            return np.asarray(self.model.get_booster().predict(dm, pred_contribs=True))
        raise TypeError(f"Unsupported tree model: {self.model_type}")

    def _native_base_value(self) -> float:
        probe = pd.DataFrame([np.zeros(len(self.feature_names))], columns=self.feature_names)
        c = self._native_contrib(probe)
        return float(c[0, -1])

    def shap_values(self, X) -> np.ndarray:
        X = self._frame(X)
        if self.explainer is not None:
            try:
                sv = self.explainer.shap_values(X)
                if isinstance(sv, list):
                    sv = sv[-1]
                sv = np.asarray(sv)
                return sv[:, :, -1] if sv.ndim == 3 else sv
            except Exception:
                pass
        c = self._native_contrib(X)
        return c[:, :-1]

    def global_importance(self, X, sensors_only: bool = False) -> pd.DataFrame:
        sv = self.shap_values(X)
        df = pd.DataFrame({
            "feature": self.feature_names,
            "mean_abs_shap": np.abs(sv).mean(0),
            "mean_shap": sv.mean(0),
        })
        if sensors_only:
            df = df[~df["feature"].map(is_indicator)]
        df = df.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
        df["rank"] = np.arange(1, len(df) + 1)
        total = float(df["mean_abs_shap"].sum())
        df["share"] = df["mean_abs_shap"] / total if total else 0.0
        return df

    def local_explanation(self, x_row: pd.Series, shap_row: np.ndarray, proba: float, top_k: int = 10) -> dict:
        order = np.argsort(-shap_row)
        pos = [i for i in order if shap_row[i] > 0][:top_k]
        neg = [i for i in order[::-1] if shap_row[i] < 0][:top_k]

        def fmt(i):
            return {
                "feature": self.feature_names[i],
                "value": float(x_row.iloc[i]),
                "contribution": float(shap_row[i]),
                "direction": "toward FAIL" if shap_row[i] > 0 else "toward PASS",
            }

        return {
            "units": "log-odds of FAIL (positive = pushes toward FAIL, negative = pushes toward PASS)",
            "base_value": self.base_value,
            "logit": float(self.base_value + shap_row.sum()),
            "failure_probability": float(proba),
            "top_positive": [fmt(i) for i in pos],
            "top_negative": [fmt(i) for i in neg],
        }
