"""2-minute demo: pick a held-out failure -> WHY -> evidence -> WHAT-IF -> best safe option."""
from __future__ import annotations

import json

import numpy as np

from src.inference.pipeline import RootCausePipeline
from src.utils import resolve


def pick_demo_sample(P: RootCausePipeline, max_tries: int = 15) -> tuple[int, dict]:
    """Highest-risk true FAIL in the TEST split that the model flags AND for which a recommendation exists."""
    test = P.splits["test"]
    pr = P.engine.model.predict_proba(P.X_model.iloc[test])[:, 1]
    cand = [int(test[i]) for i in np.argsort(-pr) if P.y.iloc[test[i]] == 1]
    flagged = [s for s in cand if P.engine.proba(P.X_model.iloc[s]) >= P.engine.thr] or cand
    best = None
    for sid in flagged[:max_tries]:
        r = P.analyze(sample_id=sid)
        best = best or (sid, r)
        if r["recommendation"]["status"] == "RECOMMENDED":
            return sid, r
    return best


def run_demo(anomaly_score=None, anomaly_label=None):
    P = RootCausePipeline()
    sid, r = pick_demo_sample(P)
    if anomaly_score is not None:
        r = P.analyze(sample_id=sid, anomaly_score=anomaly_score, anomaly_label=anomaly_label)
    p = r["prediction"]
    print(f"\n=== WHAT HAPPENED? Sample #{sid} ({r['sample']['split']}, {r['sample']['timestamp']}, actual: {r['sample']['actual_label']})")
    print(f"Predicted {p['predicted_class']}  risk {p['failure_probability']:.1%}  level {p['risk_level']}  (threshold {p['decision_threshold']:.3f})")
    print("\n=== WHY? SHAP (log-odds)")
    for t in r["explanation"]["top_positive"][:5]:
        print(f"  {t['feature']:<14} {t['contribution']:+.3f}  value={t['value']:.5g}")
    print("\n=== ROOT-CAUSE CANDIDATES (hypotheses, not proven causes)")
    print(f"{'#':<3}{'Feature':<12}{'Score':>7}  {'Conf':<7}{'σ-dev':>7}  Evidence")
    for c in r["root_cause_candidates"]:
        print(f"{c['rank']:<3}{c['feature']:<12}{c['root_cause_score']:>7.2f}  {c['confidence']:<7}{c['deviation_sigma']:>+7.1f}  {c['evidence_strength']}")
    print("\n" + r["root_cause_candidates"][0]["explanation"])
    print("\n=== SIMILAR HISTORICAL SAMPLES")
    for m in r["historical_matches"][:3]:
        print(f"  #{m['sample_id']}  {m['similarity_pct']}%  {m['label']}")
    rec = r["recommendation"]
    print("\n=== WHAT-IF / RECOMMENDATION")
    if rec["status"] == "RECOMMENDED":
        for a in rec.get("alternatives", []):
            print(f"  Option {a['option']} ({a['label']}): {a['parameter']} -> {a['value']:.5g}   risk {a['risk']:.1%}")
        print(f"\nBEST SAFE OPTION: {rec['parameter']}: {rec['current_value']} -> {rec['recommended_value']}  "
              f"risk {rec['current_risk']:.1%} -> {rec['predicted_risk']:.1%} (confidence {rec['confidence']})")
    else:
        print(" ", rec["message"])
    print("\n" + r["disclaimer"])
    (resolve(P.cfg, "reports_dir") / "example_analysis.json").write_text(json.dumps(r, indent=2))
    return r


if __name__ == "__main__":
    run_demo()
