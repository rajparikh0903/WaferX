"""Metrics, threshold selection, report writing. Accuracy is deliberately NOT a selection metric."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (average_precision_score, balanced_accuracy_score, confusion_matrix,
                             f1_score, precision_recall_curve, precision_score, recall_score,
                             roc_auc_score)

from src.utils import resolve


def compute_metrics(y, proba, thr: float) -> dict:
    y = np.asarray(y).astype(int)
    proba = np.asarray(proba, dtype=float)
    pred = (proba >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    both = len(np.unique(y)) == 2
    return {
        "threshold": float(thr),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y, proba)) if both else float("nan"),
        "pr_auc": float(average_precision_score(y, proba)) if both else float("nan"),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "n": int(len(y)), "n_fail": int(y.sum()), "prevalence": float(y.mean()),
    }


def best_threshold(y, proba) -> float:
    """Threshold maximising F1 on the given (validation) data; ties -> higher threshold."""
    y = np.asarray(y).astype(int)
    p, r, t = precision_recall_curve(y, proba)
    if len(t) == 0 or y.sum() == 0:
        return 0.5
    f1 = 2 * p[:-1] * r[:-1] / (p[:-1] + r[:-1] + 1e-12)
    best = len(f1) - 1 - int(np.argmax(f1[::-1]))
    return float(t[best])


def comparison_frame(results: dict) -> pd.DataFrame:
    rows = []
    for name, r in results.items():
        t = r["test"]
        rows.append({"Model": name, "Precision": t["precision"], "Recall": t["recall"], "F1": t["f1"],
                     "ROC-AUC": t["roc_auc"], "PR-AUC": t["pr_auc"],
                     "Balanced Accuracy": t["balanced_accuracy"],
                     "Val PR-AUC": r["val"]["pr_auc"], "Val F1": r["val"]["f1"], "Threshold": r["threshold"]})
    return pd.DataFrame(rows)


def write_evaluation_report(meta: dict, cfg: dict) -> None:
    out = resolve(cfg, "reports_dir")
    df = pd.DataFrame(meta["comparison"])
    sp = meta["split"]
    fin = meta["final_model"]
    md = [
        "# Failure-model evaluation report (REAL SECOM RESULTS)", "",
        f"Model version `{meta['model_version']}` trained {meta['training_date']}.",
        "", "## Split", "",
        f"Temporal split (chronological, no shuffling): train {sp['train']['n']} rows / {sp['train']['n_fail']} FAIL, "
        f"validation {sp['val']['n']} / {sp['val']['n_fail']}, test {sp['test']['n']} / {sp['test']['n_fail']}.",
        "Preprocessing, hyper-parameters and the decision threshold are fitted/chosen on train and validation only; "
        "the test set is scored once.",
        "", "## Model comparison (test set, threshold = validation-optimal F1)", "",
        df.round(3).to_markdown(index=False), "",
        f"Chance-level PR-AUC on the test set equals its FAIL prevalence: **{sp['test']['n_fail'] / sp['test']['n']:.3f}**.", "",
        f"## Final model: `{fin['name']}`", "",
        f"- Decision threshold: **{fin['threshold']:.4f}**",
        f"- Test: precision {fin['test']['precision']:.3f}, recall {fin['test']['recall']:.3f}, F1 {fin['test']['f1']:.3f}, "
        f"PR-AUC {fin['test']['pr_auc']:.3f}, ROC-AUC {fin['test']['roc_auc']:.3f}, "
        f"balanced accuracy {fin['test']['balanced_accuracy']:.3f}",
        f"- Test confusion matrix: TN {fin['test']['tn']}, FP {fin['test']['fp']}, FN {fin['test']['fn']}, TP {fin['test']['tp']}",
        "", "## Imputation / missing-indicator comparison (XGBoost, validation)", "",
        "```", str(meta["imputation_comparison"]), "```", "",
        "## Resampling check (validation PR-AUC)", "", "```", str(meta.get("resampling_check")), "```", "",
        "## Caveats", "",
        "- The test set has very few FAIL rows, so every test metric has a wide confidence interval; differences "
        "between models of a few points are not statistically meaningful.",
        "- SECOM fail-rate drifts over time (see data_quality_report.md); a temporal split is therefore harder, and "
        "more honest, than a random split.",
        "- Predictions for samples inside the training period are in-sample and optimistic. Demo on test-split samples.",
    ]
    (out / "model_evaluation_report.md").write_text("\n".join(md))
    df.to_csv(out / "model_comparison.csv", index=False)
