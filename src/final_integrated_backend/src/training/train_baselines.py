"""Baseline models (Logistic Regression, Random Forest, optional LightGBM) with a small validation-PR-AUC search."""
from __future__ import annotations

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.model_selection import ParameterGrid
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def search(make, grid: dict, Xtr, ytr, Xva, yva) -> dict:
    best = None
    for params in ParameterGrid(grid):
        m = make(params).fit(Xtr, ytr)
        score = average_precision_score(yva, m.predict_proba(Xva)[:, 1])
        if best is None or score > best["val_pr_auc"]:
            best = {"model": m, "params": params, "val_pr_auc": float(score)}
    return best


def fit_logreg(Xtr, ytr, Xva, yva, cfg):
    seed = cfg["project"]["seed"]
    grid = {"C": cfg["training"]["logistic_regression"]["C"]}
    make = lambda p: Pipeline([("scaler", StandardScaler()),
                               ("clf", LogisticRegression(C=p["C"], class_weight="balanced", max_iter=3000,
                                                          random_state=seed))])
    return search(make, grid, Xtr, ytr, Xva, yva)


def fit_random_forest(Xtr, ytr, Xva, yva, cfg):
    seed = cfg["project"]["seed"]
    grid = dict(cfg["training"]["random_forest"])
    make = lambda p: RandomForestClassifier(class_weight="balanced_subsample", n_jobs=-1, random_state=seed, **p)
    return search(make, grid, Xtr, ytr, Xva, yva)


def fit_lightgbm(Xtr, ytr, Xva, yva, cfg):
    t = cfg["training"]["lightgbm"]
    if not t.get("enabled", True):
        return None
    try:
        from lightgbm import LGBMClassifier
    except Exception:
        return None
    seed = cfg["project"]["seed"]
    grid = {k: v for k, v in t.items() if k != "enabled"}
    spw = float((ytr == 0).sum() / max((ytr == 1).sum(), 1))
    make = lambda p: LGBMClassifier(scale_pos_weight=spw, subsample=0.8, subsample_freq=1, min_child_samples=10,
                                    random_state=seed, verbosity=-1, n_jobs=-1, **p)
    return search(make, grid, Xtr, ytr, Xva, yva)
