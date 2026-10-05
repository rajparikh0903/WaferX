"""SYNTHETIC root-cause validation (NOT SECOM): known causal features are injected, then we check whether the
root-cause engine ranks them at the top. SECOM itself has no ground-truth causes, so none is claimed for it."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedKFold

from src.explainability.shap_engine import ShapEngine
from src.inference.pipeline import AnalysisEngine, build_reference_bundle
from src.root_cause.evidence_engine import analyze_anomaly
from src.training.evaluate import best_threshold
from src.utils import load_config, resolve, to_py


def make_synthetic(n, d, causal, seed):
    rng = np.random.default_rng(seed)
    common = rng.normal(size=(n, 1))
    Z = rng.normal(size=(n, d)) + 0.4 * common                      # mildly correlated "process measurements"
    Z = (Z - Z.mean(0)) / Z.std(0)
    a, b, c = causal
    logit = (-4.0 + 2.2 * np.maximum(Z[:, a] - 0.8, 0) + 2.2 * np.maximum(Z[:, b] - 0.8, 0)
             + 1.6 * np.maximum(Z[:, c] - 0.8, 0) + 1.0 * ((Z[:, a] > 1) & (Z[:, b] > 1)))
    y = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
    rs = np.random.default_rng(seed + 1)
    X = Z * rs.uniform(0.5, 50, d) + rs.uniform(-100, 500, d)         # arbitrary units
    names = [f"Feature {i + 1}" for i in range(d)]
    return pd.DataFrame(X, columns=names), y, Z, names


def run_synthetic(cfg=None) -> dict:
    cfg = cfg or load_config()
    s = cfg["synthetic"]
    seed = s["seed"]
    X, y, Z, names = make_synthetic(s["n_samples"], s["n_features"], s["causal_features"], seed)
    causal = [names[i] for i in s["causal_features"]]
    n = len(X)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    tr, va, te = perm[: int(.6 * n)], perm[int(.6 * n): int(.8 * n)], perm[int(.8 * n):]
    spw = float((y[tr] == 0).sum() / max(y[tr].sum(), 1))
    mk = lambda: xgb.XGBClassifier(n_estimators=250, max_depth=3, learning_rate=0.06, subsample=0.8, colsample_bytree=0.7,
                                   scale_pos_weight=spw ** 0.5, tree_method="hist", random_state=seed, n_jobs=-1,
                                   eval_metric="aucpr")
    model = mk().fit(X.iloc[tr], y[tr])
    thr = best_threshold(y[va], model.predict_proba(X.iloc[va])[:, 1])
    pr_auc = float(average_precision_score(y[te], model.predict_proba(X.iloc[te])[:, 1]))

    shap_engine = ShapEngine(model, names)
    dev = np.concatenate([tr, va])
    gi = shap_engine.global_importance(X.iloc[dev])
    oof = np.zeros(len(tr))
    for a, b in StratifiedKFold(5, shuffle=True, random_state=seed).split(X.iloc[tr], y[tr]):
        oof[b] = mk().fit(X.iloc[tr].iloc[a], y[tr][a]).predict_proba(X.iloc[tr].iloc[b])[:, 1]
    bundle = build_reference_bundle(X.iloc[tr], X.iloc[tr], y[tr], names, gi, oof, thr, None, cfg)
    eng = AnalysisEngine(model, names, thr, shap_engine, bundle, cfg)

    cidx = s["causal_features"]
    methods = {"full_root_cause_score": [], "shap_only": [], "deviation_only": []}
    n_eval = 0
    for i in te[y[te] == 1]:
        truth = {names[j] for j in cidx if Z[i, j] > 1.5}               # causes that are actually abnormal here
        if not truth:
            continue
        n_eval += 1
        x = X.iloc[i]
        r = eng.root_cause(x, anomaly=analyze_anomaly(None), top_k=cfg["root_cause"]["candidate_pool"])
        cands = r["root_cause_candidates"]
        full = [c["feature"] for c in cands]
        shap_only = [c["feature"] for c in sorted(cands, key=lambda c: -c["shap_contribution"])]
        dev_only = [c["feature"] for c in sorted(cands, key=lambda c: -abs(c["deviation_sigma"]))]
        for k, order in (("full_root_cause_score", full), ("shap_only", shap_only), ("deviation_only", dev_only)):
            ranks = [order.index(t) + 1 for t in truth if t in order]
            first = min(ranks) if ranks else None
            methods[k].append({"top1": bool(order[0] in truth), "top3": bool(set(order[:3]) & truth),
                               "rr": 1.0 / first if first else 0.0})
    summary = {}
    for k, v in methods.items():
        summary[k] = {"top1_identification": float(np.mean([m["top1"] for m in v])),
                      "top3_identification": float(np.mean([m["top3"] for m in v])),
                      "mean_reciprocal_rank": float(np.mean([m["rr"] for m in v]))}
    d = s["n_features"]
    k_truth_avg = 1.0
    out = {"LABEL": "SYNTHETIC ROOT-CAUSE VALIDATION - generated data, not SECOM",
           "setup": {"n_samples": n, "n_features": d, "injected_causal_features": causal,
                     "fail_rate": float(y.mean()), "test_failures_evaluated": n_eval,
                     "failure_model_test_pr_auc": pr_auc, "threshold": thr,
                     "truth_definition": "injected causal features with z > 1.5 in that failing sample"},
           "results": summary,
           "random_baseline_top1_approx": float(len(cidx) / d),
           "note": "Anomaly evidence was not supplied here (works without it). Results validate the ranking logic on a "
                   "problem where the answer is known; they say nothing about SECOM root-cause accuracy."}
    out = to_py(out)
    rep = resolve(cfg, "reports_dir")
    (rep / "synthetic_validation.json").write_text(json.dumps(out, indent=2))
    md = ["# SYNTHETIC ROOT-CAUSE VALIDATION (generated data - not SECOM)", "",
          f"{n} samples, {d} features, injected causes {causal}; {n_eval} test failures evaluated; "
          f"failure-model test PR-AUC {pr_auc:.3f}.", "",
          "| ranking method | Top-1 | Top-3 | MRR |", "|---|---|---|---|"]
    for k, v in out["results"].items():
        md.append(f"| {k} | {v['top1_identification']:.3f} | {v['top3_identification']:.3f} | {v['mean_reciprocal_rank']:.3f} |")
    md += ["", f"Random-guess Top-1 would be about {out['random_baseline_top1_approx']:.3f}.", "", out["note"]]
    (rep / "synthetic_validation.md").write_text("\n".join(md))
    print("\n".join(md))
    return out


if __name__ == "__main__":
    run_synthetic()
