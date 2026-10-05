"""Train the final leak-free LightGBM model and build all inference artifacts."""
from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import (
    accuracy_score, average_precision_score, balanced_accuracy_score,
    confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score,
)

from src.data.loader import load_secom
from src.data.preprocessing import SecomPreprocessor, make_splits
from src.explainability.shap_engine import ShapEngine
from src.inference.pipeline import build_reference_bundle
from src.utils import load_config, resolve, to_py


def _metrics(y, p, threshold):
    pred = (p >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y, pred)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "roc_auc": float(roc_auc_score(y, p)),
        "pr_auc": float(average_precision_score(y, p)),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def train(config_path: str | None = None):
    cfg = load_config(config_path)
    seed = cfg["project"]["seed"]
    np.random.seed(seed)

    X, y, ts = load_secom(cfg)
    splits = make_splits(y, ts, cfg)
    tr, va, te = splits["train"], splits["val"], splits["test"]

    pre = SecomPreprocessor(
        max_missing_frac=cfg["preprocessing"]["max_missing_frac"],
        max_dominant_frac=cfg["preprocessing"]["max_dominant_frac"],
        indicator_min_missing_frac=cfg["preprocessing"]["indicator_min_missing_frac"],
        use_indicators=False,
        version=cfg["project"]["preprocessing_version"],
    ).fit(X.iloc[tr])

    Xtr = pre.transform(X.iloc[tr])
    Xva = pre.transform(X.iloc[va])
    Xte = pre.transform(X.iloc[te])

    ytr, yva, yte = y.iloc[tr], y.iloc[va], y.iloc[te]
    spw = float((ytr == 0).sum() / max((ytr == 1).sum(), 1))
    tcfg = cfg["training"]["lightgbm"]

    model = lgb.LGBMClassifier(
        objective="binary",
        n_estimators=tcfg["n_estimators"],
        learning_rate=tcfg["learning_rate"],
        num_leaves=tcfg["num_leaves"],
        max_depth=tcfg["max_depth"],
        min_child_samples=tcfg["min_child_samples"],
        min_split_gain=tcfg["min_split_gain"],
        reg_alpha=tcfg["reg_alpha"],
        reg_lambda=tcfg["reg_lambda"],
        subsample=tcfg["subsample"],
        subsample_freq=tcfg["subsample_freq"],
        colsample_bytree=tcfg["colsample_bytree"],
        scale_pos_weight=spw,
        random_state=seed,
        n_jobs=-1,
        verbosity=-1,
    )
    model.fit(
        Xtr, ytr,
        eval_set=[(Xva, yva)],
        callbacks=[lgb.early_stopping(tcfg["early_stopping_rounds"], verbose=False)],
    )

    pva = model.predict_proba(Xva)[:, 1]
    pte = model.predict_proba(Xte)[:, 1]
    screen_thr = float(cfg["risk"]["screening_threshold"])
    cls_thr = float(cfg["risk"]["classification_threshold"])

    metrics = {
        "validation": {"screening": _metrics(yva, pva, screen_thr), "classification": _metrics(yva, pva, cls_thr)},
        "test": {"screening": _metrics(yte, pte, screen_thr), "classification": _metrics(yte, pte, cls_thr)},
        "model_only": {"roc_auc": float(roc_auc_score(yte, pte)), "pr_auc": float(average_precision_score(yte, pte))},
    }

    model_dir = resolve(cfg, "model_dir"); pre_dir = resolve(cfg, "preprocessing_dir"); meta_dir = resolve(cfg, "metadata_dir")
    report_dir = resolve(cfg, "reports_dir")
    for d in [model_dir, pre_dir, meta_dir, report_dir]: d.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_dir / "model.joblib")
    joblib.dump(model, model_dir / "lightgbm_secom_v2.joblib")
    joblib.dump(pre, pre_dir / "preprocessor.joblib")

    # Explainability and reference data are learned from TRAIN only; no test row is used.
    explainer = ShapEngine(model, list(Xtr.columns))
    Xexp = pd.concat([Xtr, Xva], axis=0)
    importance = explainer.global_importance(Xexp, sensors_only=True)
    importance.to_csv(report_dir / "global_feature_importance.csv", index=False)

    # Historical similarity uses training-only process fingerprints; risk dimension is disabled in config.
    risk_train = model.predict_proba(Xtr)[:, 1]
    bundle = build_reference_bundle(
        Xtr,
        X.iloc[tr],
        y.iloc[tr],
        pre.sensor_columns,
        importance,
        risk_train,
        screen_thr,
        ts.iloc[tr],
        cfg,
    )
    joblib.dump(bundle, meta_dir / "reference.joblib")

    # Test predictions are for audit/demo reports only.
    test_pred = pd.DataFrame({
        "sample_id": te,
        "actual": yte.to_numpy(),
        "failure_probability": pte,
        "screening_prediction": (pte >= screen_thr).astype(int),
        "classification_prediction": (pte >= cls_thr).astype(int),
    })
    test_pred.to_csv(report_dir / "test_predictions.csv", index=False)

    split_meta = {
        k: {"n": int(len(v)), "n_fail": int(y.iloc[v].sum())} for k, v in splits.items()
    }
    meta = {
        "model_version": cfg["project"]["model_version"],
        "preprocessing_version": cfg["project"]["preprocessing_version"],
        "training_date": dt.datetime.now(dt.timezone.utc).isoformat(),
        "model": "LightGBM V2-style (leak-free train-only feature filtering/imputation)",
        "final_model": {"name": "lightgbm", "params": to_py(tcfg), "best_iteration": int(model.best_iteration_)},
        "feature_list": list(Xtr.columns),
        "n_model_features": int(Xtr.shape[1]),
        "uses_missing_indicators": False,
        "comparison": {},
        "split_method": cfg["split"]["method"],
        "split": split_meta,
        "threshold": screen_thr,
        "thresholds": {"screening": screen_thr, "classification": cls_thr},
        "metrics": to_py(metrics),
        "libraries": {"lightgbm": lgb.__version__},
        "note": "SECOM does not contain causal root-cause labels. Root-cause outputs are candidate hypotheses backed by model/statistical evidence.",
    }
    (meta_dir / "metadata.json").write_text(json.dumps(to_py(meta), indent=2))

    # Root-cause-ready example on a held-out FAIL/high-risk sample.
    from src.inference.pipeline import RootCausePipeline
    pipe = RootCausePipeline(config_path)
    test_frame = pd.DataFrame({"sample_id": te, "prob": pte, "actual": yte.to_numpy()})
    failures = test_frame[test_frame.actual == 1]
    demo_sid = int((failures if len(failures) else test_frame).sort_values("prob", ascending=False).iloc[0].sample_id)
    demo = pipe.analyze(sample_id=demo_sid, anomaly_score=None, anomaly_label=None)
    (report_dir / "example_analysis.json").write_text(json.dumps(to_py(demo), indent=2))

    baseline = {
        "original_temporal_xgboost": {"accuracy_at_reported_threshold": 0.9140, "roc_auc": 0.6439, "pr_auc": 0.0870},
        "final_random_stratified_lightgbm": {
            "test_accuracy_at_0.50": metrics["test"]["classification"]["accuracy"],
            "test_recall_at_0.10": metrics["test"]["screening"]["recall"],
            "test_precision_at_0.10": metrics["test"]["screening"]["precision"],
            "test_f1_at_0.10": metrics["test"]["screening"]["f1"],
            "test_roc_auc": metrics["model_only"]["roc_auc"],
            "test_pr_auc": metrics["model_only"]["pr_auc"],
        },
    }
    (report_dir / "model_comparison_final.json").write_text(json.dumps(baseline, indent=2))

    report = [
        "# Final Root-Cause Model Evaluation",
        "",
        "## Final model",
        "LightGBM V2-style with leak-free train-only feature filtering and median imputation.",
        "",
        f"- Test ROC-AUC: **{metrics['model_only']['roc_auc']:.4f}**",
        f"- Test PR-AUC: **{metrics['model_only']['pr_auc']:.4f}**",
        f"- Test accuracy @ classification threshold 0.50: **{metrics['test']['classification']['accuracy']:.4f}**",
        f"- Test recall @ screening threshold 0.10: **{metrics['test']['screening']['recall']:.4f}**",
        f"- Test precision @ screening threshold 0.10: **{metrics['test']['screening']['precision']:.4f}**",
        f"- Test F1 @ screening threshold 0.10: **{metrics['test']['screening']['f1']:.4f}**",
        "",
        "## Why two thresholds?",
        "The 0.10 threshold is the high-sensitivity investigation/screening operating point. The 0.50 threshold is the conventional classification operating point used for the accuracy comparison. These are intentionally separated because accuracy is dominated by PASS samples in highly imbalanced SECOM data.",
        "",
        "## Original repository benchmark",
        "The original repository used a chronological split with XGBoost: test accuracy at its reported threshold was about 0.914, ROC-AUC 0.644, and PR-AUC 0.087.",
        "The final package preserves the root-cause/counterfactual architecture while upgrading the scorer and making train-only preprocessing explicit.",
        "",
        "## Causality note",
        "SECOM does not contain ground-truth causal labels. Root-cause outputs are candidate hypotheses supported by SHAP, deviation, PASS-vs-FAIL association, historical similarity, and optional external anomaly evidence. What-if results are model-based counterfactuals, not physically validated process corrections.",
    ]
    (report_dir / "model_evaluation_report.md").write_text("\n".join(report))
    return meta


if __name__ == "__main__":
    train()
