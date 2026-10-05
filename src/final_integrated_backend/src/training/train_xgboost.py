"""Main training script: preprocessing, baselines, XGBoost, threshold, SHAP importance, reference artefacts."""
from __future__ import annotations

import datetime as dt
import json
import platform

import joblib
import numpy as np
import pandas as pd
import sklearn
import xgboost as xgb
from sklearn.metrics import average_precision_score
from sklearn.model_selection import ParameterGrid, StratifiedKFold

from src.data.loader import load_secom
from src.data.preprocessing import SecomPreprocessor, make_splits
from src.data.validation import data_quality_report, write_quality_report
from src.training.evaluate import best_threshold, comparison_frame, compute_metrics, write_evaluation_report
from src.training.train_baselines import fit_lightgbm, fit_logreg, fit_random_forest
from src.utils import resolve, to_py

DEFAULT_XGB = dict(max_depth=3, n_estimators=300, learning_rate=0.05, colsample_bytree=0.4,
                   min_child_weight=3, spw_mult=1.0)


def make_xgb(p: dict, y_train, cfg) -> xgb.XGBClassifier:
    spw = float((y_train == 0).sum() / max((y_train == 1).sum(), 1)) * p.get("spw_mult", 1.0)
    q = {k: v for k, v in p.items() if k != "spw_mult"}
    return xgb.XGBClassifier(objective="binary:logistic", eval_metric="aucpr", tree_method="hist",
                             subsample=cfg["training"]["xgboost"]["subsample"], scale_pos_weight=spw,
                             random_state=cfg["project"]["seed"], n_jobs=-1, **q)


def fit_xgboost(Xtr, ytr, Xva, yva, cfg) -> dict:
    grid = {k: v for k, v in cfg["training"]["xgboost"].items() if k != "subsample"}
    best = None
    for params in ParameterGrid(grid):
        m = make_xgb(params, ytr, cfg).fit(Xtr, ytr)
        s = average_precision_score(yva, m.predict_proba(Xva)[:, 1])
        if best is None or s > best["val_pr_auc"]:
            best = {"model": m, "params": params, "val_pr_auc": float(s)}
    return best


def _evaluate(name, model, data, extra=None) -> dict:
    (Xva, yva), (Xte, yte) = data["val"], data["test"]
    pva, pte = model.predict_proba(Xva)[:, 1], model.predict_proba(Xte)[:, 1]
    thr = best_threshold(yva, pva)
    return {"name": name, "threshold": thr, "val": compute_metrics(yva, pva, thr),
            "test": compute_metrics(yte, pte, thr), "proba_val": pva, "proba_test": pte, **(extra or {})}


def run_training(cfg: dict) -> dict:
    from src.inference.pipeline import build_reference_bundle  # late import (avoids cycles)
    from src.explainability.shap_engine import ShapEngine

    seed = cfg["project"]["seed"]
    np.random.seed(seed)
    X, y, ts = load_secom(cfg)
    rep = data_quality_report(X, y, ts, cfg)
    write_quality_report(rep, cfg)
    splits = make_splits(y, ts, cfg)
    (resolve(cfg, "processed_dir") / "splits.json").write_text(
        json.dumps({k: v.tolist() for k, v in splits.items()}))
    tr, va, te = splits["train"], splits["val"], splits["test"]

    # ---- preprocessing (fit on train only) ---------------------------------------------------
    pc = cfg["preprocessing"]
    pre = SecomPreprocessor(pc["max_missing_frac"], pc["max_dominant_frac"], pc["indicator_min_missing_frac"],
                            use_indicators=True, version=cfg["project"]["preprocessing_version"]).fit(X.iloc[tr])
    mats = {k: {"with": pre.transform(X.iloc[i], with_indicators=True),
                "without": pre.transform(X.iloc[i], with_indicators=False)}
            for k, i in splits.items()}
    ys = {k: y.iloc[i].to_numpy() for k, i in splits.items()}

    # ---- imputation / missing-indicator comparison (XGBoost, default params, validation PR-AUC) -
    cmp = {}
    for key in ("without", "with"):
        m = make_xgb(DEFAULT_XGB, ys["train"], cfg).fit(mats["train"][key], ys["train"])
        cmp[f"median_imputation{'+indicators' if key == 'with' else ''}"] = float(
            average_precision_score(ys["val"], m.predict_proba(mats["val"][key])[:, 1]))
    mode = str(pc["missing_indicators"]).lower()
    if mode == "auto":
        use_ind = cmp["median_imputation+indicators"] > cmp["median_imputation"] + 0.01
        cmp["decision"] = ("median+indicators (val PR-AUC better by >0.01)" if use_ind
                           else "median only (indicators did not improve val PR-AUC by >0.01; simpler wins)")
    else:
        use_ind = mode == "true"
        cmp["decision"] = f"forced by config ({mode})"
    pre.use_indicators = use_ind
    key = "with" if use_ind else "without"
    data = {k: (mats[k][key], ys[k]) for k in ("train", "val", "test")}
    Xtr, ytr = data["train"]
    Xva, yva = data["val"]
    print("imputation comparison:", cmp)

    # ---- models ------------------------------------------------------------------------------
    fitted = {"logistic_regression": fit_logreg(Xtr, ytr, Xva, yva, cfg),
              "random_forest": fit_random_forest(Xtr, ytr, Xva, yva, cfg),
              "xgboost": fit_xgboost(Xtr, ytr, Xva, yva, cfg)}
    lgb = fit_lightgbm(Xtr, ytr, Xva, yva, cfg)
    if lgb is not None:
        fitted["lightgbm"] = lgb
    results = {}
    for name, f in fitted.items():
        results[name] = _evaluate(name, f["model"], data, {"params": f["params"]})
        t = results[name]["test"]
        print(f"{name:20s} valPR={results[name]['val']['pr_auc']:.3f} testPR={t['pr_auc']:.3f} "
              f"F1={t['f1']:.3f} P={t['precision']:.3f} R={t['recall']:.3f}")

    # ---- SMOTE vs class weights (validation PR-AUC) --------------------------------------------
    resampling = {"class_weights (scale_pos_weight)": fitted["xgboost"]["val_pr_auc"]}
    if cfg["training"].get("resampling_check"):
        try:
            from imblearn.over_sampling import SMOTE
            Xs, ys_ = SMOTE(random_state=seed, k_neighbors=min(5, int(ytr.sum()) - 1)).fit_resample(Xtr, ytr)
            m = make_xgb(fitted["xgboost"]["params"], ytr, cfg)
            m.set_params(scale_pos_weight=1.0)       # classes are balanced by SMOTE instead
            m.fit(Xs, ys_)
            resampling["SMOTE (scale_pos_weight=1)"] = float(average_precision_score(yva, m.predict_proba(Xva)[:, 1]))
            resampling["decision"] = ("class weights kept" if resampling["SMOTE (scale_pos_weight=1)"]
                                      <= resampling["class_weights (scale_pos_weight)"] else "SMOTE scored higher on validation but "
                                      "class weights were kept for a simpler, leak-free pipeline")
        except ImportError:
            resampling["note"] = "imbalanced-learn not installed; SMOTE not evaluated"

    # ---- final model ---------------------------------------------------------------------------
    final_name = cfg["training"]["final_model"]
    if final_name not in ("xgboost", "random_forest", "lightgbm"):
        raise ValueError("final_model must be a tree model for SHAP-based root cause analysis")
    final = results[final_name]
    model = fitted[final_name]["model"]
    best_by_val = max(results, key=lambda k: results[k]["val"]["pr_auc"])
    model_features = list(Xtr.columns)

    # ---- explainability + reference artefacts (train-only statistics) ---------------------------
    shap_engine = ShapEngine(model, model_features)
    dev_idx = np.concatenate([tr, va])
    X_dev = pre.transform(X.iloc[dev_idx])
    gi = shap_engine.global_importance(X_dev)
    gi.to_csv(resolve(cfg, "reports_dir") / "global_feature_importance.csv", index=False)

    oof = np.zeros(len(tr))
    skf = StratifiedKFold(cfg["training"]["oof_folds"], shuffle=True, random_state=seed)
    for a, b in skf.split(Xtr, ytr):
        mm = make_xgb(fitted["xgboost"]["params"], ytr[a], cfg).fit(Xtr.iloc[a], ytr[a])
        oof[b] = mm.predict_proba(Xtr.iloc[b])[:, 1]
    bundle = build_reference_bundle(
        X_model=Xtr, X_raw=pre.raw_sensors(X.iloc[tr]), y=ytr, sensor_cols=pre.sensor_columns,
        importance=gi, risk=oof, threshold=final["threshold"], timestamps=ts.iloc[tr], cfg=cfg)

    # ---- persist --------------------------------------------------------------------------------
    pre_dir, mdl_dir, meta_dir = resolve(cfg, "preprocessing_dir"), resolve(cfg, "model_dir"), resolve(cfg, "metadata_dir")
    joblib.dump(pre, pre_dir / "preprocessor.joblib")
    joblib.dump(model, mdl_dir / "model.joblib")
    if isinstance(model, xgb.XGBClassifier):
        model.save_model(str(mdl_dir / "model.json"))
    joblib.dump(bundle, meta_dir / "reference.joblib")
    joblib.dump({"y_val": yva, "y_test": data["test"][1],
                 "proba_val": {k: v["proba_val"] for k, v in results.items()},
                 "proba_test": {k: v["proba_test"] for k, v in results.items()},
                 "thresholds": {k: v["threshold"] for k, v in results.items()}}, meta_dir / "eval_cache.joblib")

    sp = {k: {"n": int(len(i)), "n_fail": int(y.iloc[i].sum()), "from": ts.iloc[i].min(), "to": ts.iloc[i].max()}
          for k, i in splits.items()}
    strip = lambda r: {k: v for k, v in r.items() if not k.startswith("proba")}
    meta = to_py({
        "model_version": cfg["project"]["model_version"],
        "training_date": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "preprocessing_version": pre.version,
        "seed": seed,
        "split": sp, "split_method": cfg["split"]["method"],
        "final_model": {**strip(final), "params": fitted[final_name]["params"]},
        "best_model_by_val_pr_auc": best_by_val,
        "threshold": final["threshold"],
        "metrics": {"validation": final["val"], "test": final["test"]},
        "feature_list": model_features,
        "n_model_features": len(model_features),
        "sensor_columns": pre.sensor_columns,
        "uses_missing_indicators": use_ind,
        "preprocessing_report": {k: (v if not isinstance(v, list) else {"count": len(v), "names": v[:10]})
                                 for k, v in pre.report_.items()},
        "imputation_comparison": cmp,
        "resampling_check": resampling,
        "comparison": comparison_frame({k: strip(v) for k, v in results.items()}).to_dict("records"),
        "all_models": {k: strip(v) for k, v in results.items()},
        "libraries": {"python": platform.python_version(), "xgboost": xgb.__version__,
                      "sklearn": sklearn.__version__, "pandas": pd.__version__},
        "risk_cfg": cfg["risk"],
    })
    (meta_dir / "metadata.json").write_text(json.dumps(meta, indent=2))
    write_evaluation_report(meta, cfg)
    print("saved artefacts; final model:", final_name, "threshold", round(final["threshold"], 4))
    return meta
