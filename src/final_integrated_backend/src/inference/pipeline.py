"""High-level inference pipeline.

AnalysisEngine     : dataset-agnostic (model + reference stats) - reused by the real pipeline, tests and synthetic validation
RootCausePipeline  : loads saved SECOM artefacts + data and exposes analyze_sample / predict / root_cause / what_if
"""
from __future__ import annotations

import json

import joblib
import numpy as np
import pandas as pd

from src.counterfactual.constraints import build_constraints
from src.counterfactual.generator import candidate_values, predict_with_values
from src.counterfactual.optimizer import build_recommendation, optimize_feature, optimize_pair
from src.data.loader import load_secom
from src.data.preprocessing import make_splits
from src.explainability.shap_engine import ShapEngine
from src.inference.anomaly import SECOMAnomalyDetector
from src.features.feature_engineering import is_indicator
from src.root_cause.candidate_engine import RootCauseEngine
from src.root_cause.evidence_engine import ReferenceStats, analyze_anomaly
from src.similarity.similar_cases import SimilarityIndex
from src.utils import CAUSALITY_NOTE, DISCLAIMER, load_config, resolve, to_py


class ReferenceBundle:
    def __init__(self, ref, constraints, similarity, importance, threshold):
        self.ref, self.constraints, self.similarity = ref, constraints, similarity
        self.importance, self.threshold = importance, threshold


def build_reference_bundle(X_model, X_raw, y, sensor_cols, importance, risk, threshold, timestamps, cfg) -> ReferenceBundle:
    ref = ReferenceStats(X_model, X_raw, y, sensor_cols, cfg["root_cause"])
    cons = build_constraints(X_raw, y, sensor_cols, cfg["counterfactual"])
    fp = [f for f in importance["feature"] if not is_indicator(f) and f in sensor_cols][:cfg["similarity"]["fingerprint_features"]]
    sim = SimilarityIndex(ref, fp, risk, X_model.index.to_numpy(), y, None if timestamps is None else timestamps.to_numpy(),
                          cfg["similarity"], threshold)
    return ReferenceBundle(ref, cons, sim, importance, threshold)


def risk_level(p: float, thr: float, cfg: dict) -> str:
    r = cfg["risk"]
    if p < r["medium_fraction"] * thr:
        return "LOW"
    if p < thr:
        return "MEDIUM"
    return "HIGH" if p < thr + r["critical_position"] * (1 - thr) else "CRITICAL"


class AnalysisEngine:
    def __init__(self, model, model_features, threshold, shap_engine, bundle: ReferenceBundle, cfg: dict):
        self.model, self.features, self.thr = model, list(model_features), float(threshold)
        self.shap, self.bundle, self.cfg = shap_engine, bundle, cfg
        self.sensor_cols = bundle.ref.sensor_cols
        self.rc = RootCauseEngine(bundle.ref, bundle.similarity, cfg)

    def proba(self, x: pd.Series) -> float:
        X = pd.DataFrame([x[self.features].to_numpy(float)], columns=self.features)
        return float(self.model.predict_proba(X)[0, 1])

    def prediction(self, x: pd.Series, sample_id=None) -> dict:
        p = self.proba(x)
        return {"sample_id": sample_id, "predicted_class": "FAIL" if p >= self.thr else "PASS",
                "failure_probability": p, "risk_level": risk_level(p, self.thr, self.cfg),
                "decision_threshold": self.thr}

    def root_cause(self, x, sample_id=None, anomaly=None, top_k=None, exclude_id=None) -> dict:
        pred = self.prediction(x, sample_id)
        shap_row = self.shap.shap_values(x[self.features].to_frame().T)[0]
        anomaly = anomaly or analyze_anomaly(sample_id)
        cands = self.rc.rank(x, shap_row, self.features, pred["failure_probability"], anomaly, top_k, exclude_id)
        z = self.bundle.ref.robust_z(x)
        matches = self.bundle.similarity.similar(z, pred["failure_probability"], self.cfg["similarity"]["k_display"],
                                                 exclude_id=exclude_id)
        for m in matches:
            m.pop("row", None)
        return {"prediction": pred, "anomaly": anomaly,
                "explanation": self.shap.local_explanation(x[self.features], shap_row, pred["failure_probability"]),
                "root_cause_candidates": cands, "historical_matches": matches, "_shap_row": shap_row}

    def counterfactuals(self, x, candidates, current_risk) -> tuple[list, dict, list]:
        cf, ct = self.cfg["counterfactual"], self.bundle.constraints
        feats = [c["feature"] for c in candidates][:cf["top_n_features"]]
        singles = [optimize_feature(self.model, x, self.features, f, ct, current_risk, self.thr, cf) for f in feats]
        pairs = []
        for fa, fb in __import__("itertools").combinations(feats[:cf["pair_top_n"]], 2):
            r = optimize_pair(self.model, x, self.features, fa, fb, ct, current_risk, self.thr, cf)
            if r:
                pairs.append(r)
        return singles, build_recommendation(singles, pairs, cf), pairs

    def analyze(self, x, sample_meta=None, anomaly=None, top_k=None, exclude_id=None) -> dict:
        sid = (sample_meta or {}).get("sample_id")
        rc = self.root_cause(x, sid, anomaly, top_k, exclude_id)
        singles, rec, pairs = self.counterfactuals(x, rc["root_cause_candidates"], rc["prediction"]["failure_probability"])
        rc.pop("_shap_row")
        return to_py({"sample": sample_meta or {}, **rc, "counterfactuals": singles,
                      "multi_parameter_experiments": pairs, "recommendation": rec,
                      "causality_note": CAUSALITY_NOTE, "disclaimer": DISCLAIMER})

    def what_if(self, x, feature: str, values=None) -> dict:
        if feature not in self.sensor_cols:
            raise ValueError(f"Unknown or unusable feature '{feature}'")
        cf, c = self.cfg["counterfactual"], self.bundle.constraints.loc[feature]
        v, p0 = float(x[feature]), self.proba(x)
        from src.counterfactual.constraints import intervention_scale
        scale = intervention_scale(v, c["span"], cf["scale_span_floor"])
        allowed = candidate_values(v, c, cf)
        vals = np.array(sorted(set(float(a) for a in values))) if values is not None else np.append(allowed, v)
        if len(vals) == 0:
            raise ValueError("no values to evaluate")
        risks = predict_with_values(self.model, x, self.features, {feature: vals})
        allowed_set = {round(float(a), 10) for a in allowed}
        curve = [{"value": float(a), "risk": float(r), "risk_change": float(r - p0),
                  "intervention_magnitude": float(abs(a - v) / scale), "is_current": bool(abs(a - v) < 1e-12),
                  "within_constraints": bool(round(float(a), 10) in allowed_set or abs(a - v) < 1e-12)}
                 for a, r in sorted(zip(vals, risks))]
        return {"feature": feature, "current_value": v, "current_risk": p0, "decision_threshold": self.thr,
                "constraints": {"operating_range": [float(c["op_lo"]), float(c["op_hi"])],
                                "max_change_pct": cf["max_change_pct"], "max_change_abs": float(cf["max_change_pct"] * scale)},
                "curve": curve, "disclaimer": DISCLAIMER}


class RootCausePipeline:
    def __init__(self, config_path=None):
        self.cfg = cfg = load_config(config_path)
        self.pre = joblib.load(resolve(cfg, "preprocessing_dir") / "preprocessor.joblib")
        self.model = joblib.load(resolve(cfg, "model_dir") / "model.joblib")
        meta_dir = resolve(cfg, "metadata_dir")
        self.meta = json.loads((meta_dir / "metadata.json").read_text())
        self.bundle = joblib.load(meta_dir / "reference.joblib")
        self.X, self.y, self.ts = load_secom(cfg)
        self.splits = make_splits(self.y, self.ts, cfg)
        self.split_of = {int(i): k for k, idx in self.splits.items() for i in idx}
        self.X_model = self.pre.transform(self.X)
        self.features = list(self.X_model.columns)
        if self.features != self.meta["feature_list"]:
            raise RuntimeError("Feature list in metadata does not match preprocessor output")
        self.anomaly_detector = SECOMAnomalyDetector()
        self.engine = AnalysisEngine(self.model, self.features, self.meta["threshold"],
                                     ShapEngine(self.model, self.features), self.bundle, cfg)

    # -- input handling ------------------------------------------------------------------------------
    def _resolve(self, sample_id=None, features: dict | None = None):
        if (sample_id is None) == (features is None):
            raise ValueError("provide exactly one of sample_id or features")
        if sample_id is not None:
            if isinstance(sample_id, bool) or not isinstance(sample_id, (int, np.integer)):
                raise ValueError("sample_id must be an integer")
            if not 0 <= int(sample_id) < len(self.X):
                raise KeyError(f"sample_id {sample_id} not found (valid range 0..{len(self.X) - 1})")
            sid = int(sample_id)
            split = self.split_of[sid]
            meta = {"sample_id": sid, "split": split, "timestamp": self.ts.iloc[sid],
                    "actual_label": "FAIL" if self.y.iloc[sid] == 1 else "PASS",
                    "n_model_features": len(self.features),
                    "note": ("In-sample: this row was used to train the model, so its risk is optimistic; "
                             "use validation/test samples for honest demos.") if split == "train" else None}
            return self.X_model.iloc[sid], meta, sid
        bad = [k for k in features if k not in self.pre.input_columns_]
        if bad:
            raise ValueError(f"Unknown feature names: {bad[:5]}")
        row = pd.DataFrame([{c: np.nan for c in self.pre.input_columns_}])
        for k, v in features.items():
            row.loc[0, k] = np.nan if v is None else v
        x = self.pre.transform(row).iloc[0]
        return x, {"sample_id": None, "split": "external", "n_model_features": len(self.features),
                   "n_features_supplied": len(features)}, None

    def _resolve_anomaly(self, x, sid=None, anomaly_score=None, anomaly_label=None) -> dict:
        """Return external override when supplied; otherwise run the real detector."""
        if anomaly_score is not None or anomaly_label is not None:
            return analyze_anomaly(sid, anomaly_score, anomaly_label)
        return self.anomaly_detector.analyze(x.to_frame().T, sample_id=sid)

    def anomaly(self, sample_id=None, features=None) -> dict:
        x, _, sid = self._resolve(sample_id, features)
        return to_py(self.anomaly_detector.analyze(x.to_frame().T, sample_id=sid))

    # -- public API ----------------------------------------------------------------------------------
    def predict(self, sample_id=None, features=None) -> dict:
        x, meta, sid = self._resolve(sample_id, features)
        return to_py({"sample": meta, "prediction": self.engine.prediction(x, sid)})

    def root_cause(self, sample_id=None, features=None, anomaly_score=None, anomaly_label=None, top_k=None) -> dict:
        x, meta, sid = self._resolve(sample_id, features)
        an = self._resolve_anomaly(x, sid, anomaly_score, anomaly_label)
        r = self.engine.root_cause(x, sid, an, top_k, exclude_id=sid)
        r.pop("_shap_row")
        return to_py({"sample": meta, **r, "causality_note": CAUSALITY_NOTE})

    def what_if(self, feature: str, sample_id=None, features=None, values=None) -> dict:
        x, meta, sid = self._resolve(sample_id, features)
        return to_py({"sample": meta, **self.engine.what_if(x, feature, values)})

    def analyze(self, sample_id=None, features=None, anomaly_score=None, anomaly_label=None, top_k=None) -> dict:
        x, meta, sid = self._resolve(sample_id, features)
        an = self._resolve_anomaly(x, sid, anomaly_score, anomaly_label)
        return self.engine.analyze(x, meta, an, top_k, exclude_id=sid)

    def model_info(self) -> dict:
        m = self.meta
        return to_py({
            "model_version": m.get("model_version"),
            "training_date": m.get("training_date"),
            "preprocessing_version": m.get("preprocessing_version"),
            "split_method": m.get("split_method"),
            "split": m.get("split"),
            "threshold": m.get("threshold"),
            "thresholds": m.get("thresholds", {"screening": m.get("threshold")}),
            "metrics": m.get("metrics"),
            "n_model_features": m.get("n_model_features"),
            "uses_missing_indicators": m.get("uses_missing_indicators", False),
            "comparison": m.get("comparison", {}),
            "risk_cfg": self.cfg.get("risk", {}),
            "libraries": m.get("libraries", {}),
            "model": m["final_model"]["name"],
            "params": m["final_model"]["params"],
            "root_cause_weights": self.cfg["root_cause"]["weights"],
            "anomaly_detection": self.anomaly_detector.info(),
            "causality_note": CAUSALITY_NOTE,
        })

    def feature_importance(self, top_n: int = 20) -> dict:
        gi = self.bundle.importance
        gi = gi[~gi["feature"].map(is_indicator)].head(top_n)
        return to_py({"units": "mean |SHAP| in log-odds of FAIL (train+validation rows)",
                      "features": gi[["rank", "feature", "mean_abs_shap", "mean_shap", "share"]].to_dict("records")})


_DEFAULT: RootCausePipeline | None = None


def analyze_sample(sample_id: int, anomaly_score=None, anomaly_label=None, config_path=None) -> dict:
    """One-call entry point used by scripts / frontend glue."""
    global _DEFAULT
    if _DEFAULT is None:
        _DEFAULT = RootCausePipeline(config_path)
    return _DEFAULT.analyze(sample_id=sample_id, anomaly_score=anomaly_score, anomaly_label=anomaly_label)
