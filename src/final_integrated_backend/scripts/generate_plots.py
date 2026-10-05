"""Static figures for the report / frontend mock-ups (reports/figures)."""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import joblib
import shap
from sklearn.metrics import ConfusionMatrixDisplay, PrecisionRecallDisplay, RocCurveDisplay

from scripts.demo import pick_demo_sample
from src.inference.pipeline import RootCausePipeline
from src.utils import resolve


def generate_all():
    P = RootCausePipeline()
    cfg = P.cfg
    fig_dir = resolve(cfg, "reports_dir") / "figures"
    fig_dir.mkdir(exist_ok=True)
    ev = joblib.load(resolve(cfg, "metadata_dir") / "eval_cache.joblib")
    name = P.meta["final_model"]["name"]
    save = lambda n: (plt.tight_layout(), plt.savefig(fig_dir / n, dpi=130), plt.close("all"))

    # ---- model ----
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ConfusionMatrixDisplay.from_predictions(ev["y_test"], (ev["proba_test"][name] >= ev["thresholds"][name]).astype(int),
                                            display_labels=["PASS", "FAIL"], ax=ax, colorbar=False)
    ax.set_title(f"Confusion matrix - {name} (test)"); save("confusion_matrix.png")
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    for k, p in ev["proba_test"].items():
        PrecisionRecallDisplay.from_predictions(ev["y_test"], p, name=k, ax=ax)
    ax.axhline(ev["y_test"].mean(), ls="--", c="grey", label="chance"); ax.legend(fontsize=8); ax.set_title("PR curves (test)"); save("pr_curve.png")
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    for k, p in ev["proba_test"].items():
        RocCurveDisplay.from_predictions(ev["y_test"], p, name=k, ax=ax)
    ax.plot([0, 1], [0, 1], "--", c="grey"); ax.legend(fontsize=8); ax.set_title("ROC curves (test)"); save("roc_curve.png")
    import pandas as pd
    df = pd.DataFrame(P.meta["comparison"]).set_index("Model")[["Precision", "Recall", "F1", "PR-AUC", "ROC-AUC"]]
    df.plot.bar(figsize=(8, 4)); plt.title("Model comparison (test, temporal split)"); plt.xticks(rotation=20); save("model_comparison.png")

    # ---- SHAP ----
    dev = np.concatenate([P.splits["train"], P.splits["val"]])
    Xd = P.X_model.iloc[dev]
    sv = P.engine.shap.shap_values(Xd)
    shap.summary_plot(sv, Xd, max_display=20, show=False); save("shap_summary.png")
    shap.summary_plot(sv, Xd, plot_type="bar", max_display=20, show=False); save("shap_bar.png")

    sid, r = pick_demo_sample(P)
    x = P.X_model.iloc[sid]
    row = P.engine.shap.shap_values(x.to_frame().T)[0]
    shap.plots.waterfall(shap.Explanation(values=row, base_values=P.engine.shap.base_value, data=x.to_numpy(),
                                          feature_names=P.features), max_display=12, show=False)
    plt.title(f"SHAP waterfall - sample #{sid}"); save("shap_waterfall.png")

    # ---- root cause ----
    cands = r["root_cause_candidates"]
    names = [c["feature"] for c in cands][::-1]
    fig, ax = plt.subplots(figsize=(7, 3.8))
    left = np.zeros(len(cands))
    for comp in ["shap", "deviation", "failure_association", "historical", "anomaly"]:
        w = np.array([(c["weights_used"].get(comp, 0) * (c["components"][comp] or 0)) for c in cands])[::-1]
        ax.barh(names, w, left=left, label=comp); left += w
    ax.set_xlabel("Root-cause candidate score (weighted evidence)"); ax.legend(fontsize=8, loc="lower right")
    ax.set_title(f"Top root-cause candidates - sample #{sid}"); save("root_cause_candidates.png")
    for c in cands[:3]:
        h = c["comparison"]["histogram"]
        if not h:
            continue
        e = np.array(h["edges"]); mid = (e[:-1] + e[1:]) / 2
        fig, ax = plt.subplots(figsize=(6, 3.6))
        ax.step(mid, h["pass_fraction"], where="mid", label="PASS"); ax.step(mid, h["fail_fraction"], where="mid", label="FAIL")
        ax.axvline(c["current_value"], c="red", ls="--", label=f"current {c['current_value']:.4g}")
        ax.set_title(f"{c['feature']}: PASS vs FAIL vs current"); ax.legend(); save(f"distribution_{c['feature'].replace(' ', '_')}.png")

    # ---- what-if ----
    feat = r["recommendation"].get("parameter") if r["recommendation"]["status"] == "RECOMMENDED" else cands[0]["feature"]
    feat = feat.split(" + ")[0]
    w = P.what_if(feat, sample_id=sid)
    cv = w["curve"]
    fig, ax = plt.subplots(figsize=(6, 3.8))
    ax.plot([c["value"] for c in cv], [c["risk"] for c in cv], marker="o", ms=3)
    ax.axhline(w["decision_threshold"], c="grey", ls="--", label="decision threshold")
    ax.axvline(w["current_value"], c="red", ls=":", label="current"); ax.legend()
    ax.set_xlabel(f"{feat} (hypothetical value)"); ax.set_ylabel("predicted failure risk")
    ax.set_title("What-if curve (model-based, not physically validated)"); save("what_if_curve.png")
    print("figures written to", fig_dir)


if __name__ == "__main__":
    generate_all()
