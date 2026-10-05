"""Safe intervention optimiser.

    objective = predicted_failure_risk + lambda * intervention_magnitude
    intervention_magnitude = |new - current| / max(|current|, P5-P95 span)

A recommendation must (a) stay inside max change + operating range, (b) reduce risk by at least
min_abs_reduction, and (c) minimise the objective - the "safest reasonable" option, not the lowest-risk one.
"""
from __future__ import annotations

import itertools

import numpy as np
import pandas as pd

from src.counterfactual.constraints import intervention_scale
from src.counterfactual.generator import candidate_values, predict_with_values
from src.utils import DISCLAIMER


def _confidence(crosses, robust, within_pass, rel_red, current_above=True) -> tuple[str, dict]:
    f = {"crosses_decision_threshold": bool(crosses), "locally_robust": bool(robust),
         "within_pass_range": bool(within_pass), "relative_reduction_ge_25pct": bool(rel_red >= 0.25)}
    n = sum(f.values())
    conf = "high" if n >= 3 else "medium" if n == 2 else "low"
    if conf == "high" and not crosses and current_above:
        conf = "medium"            # still predicted FAIL after the change -> never "high"
    return conf, f


def optimize_feature(model, x_row, model_features, feature, ctable, current_risk, threshold, cf) -> dict:
    c, v = ctable.loc[feature], float(x_row[feature])
    scale = intervention_scale(v, c["span"], cf["scale_span_floor"])
    lam = cf["lambda_penalty"]
    base = {"parameter": feature, "current_value": v, "current_risk": float(current_risk),
            "operating_range": [float(c["op_lo"]), float(c["op_hi"])],
            "max_change_abs": float(cf["max_change_pct"] * scale), "disclaimer": DISCLAIMER}
    empty = {"recommended_value": None, "predicted_risk": None, "risk_reduction": None, "relative_reduction": None,
             "intervention_magnitude": None, "confidence": None, "alternatives": [], "curve": []}
    cands = candidate_values(v, c, cf)
    if len(cands) == 0:
        return {**base, **empty, "status": "NO_FEASIBLE_RANGE",
                "message": "Current value lies outside the allowed operating range by more than the maximum change."}
    risks = predict_with_values(model, x_row, model_features, {feature: cands})
    mags = np.abs(cands - v) / scale
    obj, red = risks + lam * mags, current_risk - risks
    curve = [{"value": float(a), "risk": float(r), "risk_reduction": float(d), "intervention_magnitude": float(m),
              "objective": float(o)} for a, r, d, m, o in zip(cands, risks, red, mags, obj)]
    elig = np.where(red >= cf["min_abs_reduction"])[0]
    if len(elig) == 0:
        return {**base, **empty, "curve": curve, "status": "NO_EFFECTIVE_INTERVENTION",
                "message": f"No single change within limits reduces predicted risk by >= {cf['min_abs_reduction']:.2f}.",
                "lowest_risk_found": float(risks.min())}
    ib = int(elig[np.argmin(obj[elig])])
    imin = int(elig[np.argmin(mags[elig] + 1e-6 * risks[elig])])
    iagg = int(np.argmin(risks))
    labels: dict[int, list[str]] = {}
    for i, name in ((imin, "minimal change"), (ib, "balanced (recommended)"), (iagg, "lowest risk")):
        labels.setdefault(i, []).append(name)
    alts = []
    for k, i in enumerate(sorted(labels, key=lambda i: mags[i])):
        alts.append({"option": "ABC"[k], "label": " / ".join(labels[i]), "parameter": feature,
                     "value": float(cands[i]), "risk": float(risks[i]), "risk_reduction": float(red[i]),
                     "intervention_magnitude": float(mags[i]), "objective": float(obj[i]), "recommended": i == ib})
    crosses = bool(current_risk >= threshold > risks[ib])
    nb = [j for j in (ib - 1, ib + 1) if 0 <= j < len(cands)]
    robust = all(red[j] >= cf["robust_neighbour_fraction"] * red[ib] for j in nb)
    within = bool(c["pass_p5"] <= cands[ib] <= c["pass_p95"])
    rel = float(red[ib] / current_risk) if current_risk > 0 else 0.0
    conf, factors = _confidence(crosses, robust, within, rel, current_risk >= threshold)
    return {**base, "status": "RECOMMENDED", "recommended_value": float(cands[ib]),
            "predicted_risk": float(risks[ib]), "risk_reduction": float(red[ib]), "relative_reduction": rel,
            "intervention_magnitude": float(mags[ib]), "objective": float(obj[ib]), "confidence": conf,
            "confidence_factors": factors, "crosses_decision_threshold": crosses, "within_pass_range": within,
            "alternatives": alts, "curve": curve}


def _subsample(vals: np.ndarray, v: float, n: int) -> np.ndarray:
    if len(vals) > n:
        vals = vals[np.unique(np.linspace(0, len(vals) - 1, n).round().astype(int))]
    return vals


def optimize_pair(model, x_row, model_features, fa, fb, ctable, current_risk, threshold, cf) -> dict | None:
    """Controlled two-feature experiment on a coarse grid (both features changed)."""
    ca, cb = ctable.loc[fa], ctable.loc[fb]
    va, vb = float(x_row[fa]), float(x_row[fb])
    A = _subsample(candidate_values(va, ca, cf), va, cf["pair_grid_points"])
    B = _subsample(candidate_values(vb, cb, cf), vb, cf["pair_grid_points"])
    if len(A) == 0 or len(B) == 0:
        return None
    pairs = np.array(list(itertools.product(A, B)))
    risks = predict_with_values(model, x_row, model_features, {fa: pairs[:, 0], fb: pairs[:, 1]})
    ma = np.abs(pairs[:, 0] - va) / intervention_scale(va, ca["span"], cf["scale_span_floor"])
    mb = np.abs(pairs[:, 1] - vb) / intervention_scale(vb, cb["span"], cf["scale_span_floor"])
    obj, red = risks + cf["lambda_penalty"] * (ma + mb), current_risk - risks
    elig = np.where(red >= cf["min_abs_reduction"])[0]
    if len(elig) == 0:
        return None
    i = int(elig[np.argmin(obj[elig])])
    return {"type": "pair", "parameter": f"{fa} + {fb}",
            "changes": [{"parameter": fa, "current_value": va, "recommended_value": float(pairs[i, 0])},
                        {"parameter": fb, "current_value": vb, "recommended_value": float(pairs[i, 1])}],
            "current_risk": float(current_risk), "predicted_risk": float(risks[i]), "risk_reduction": float(red[i]),
            "relative_reduction": float(red[i] / current_risk) if current_risk > 0 else 0.0,
            "intervention_magnitude": float(ma[i] + mb[i]), "objective": float(obj[i]),
            "crosses_decision_threshold": bool(current_risk >= threshold > risks[i]),
            "grid_points_evaluated": int(len(pairs))}


def build_recommendation(singles: list[dict], pairs: list[dict], cf: dict) -> dict:
    ok = [s for s in singles if s["status"] == "RECOMMENDED"]
    best = min(ok, key=lambda s: s["objective"]) if ok else None
    best_pair = min(pairs, key=lambda p: p["objective"]) if pairs else None
    if best_pair and (best is None or best_pair["objective"] < best["objective"] - cf["pair_min_improvement"]):
        return {"status": "RECOMMENDED", **best_pair, "confidence": "medium" if best_pair["crosses_decision_threshold"] else "low",
                "disclaimer": DISCLAIMER}
    if best:
        keys = ["parameter", "current_value", "recommended_value", "current_risk", "predicted_risk", "risk_reduction",
                "relative_reduction", "intervention_magnitude", "objective", "confidence", "confidence_factors",
                "crosses_decision_threshold", "within_pass_range", "alternatives"]
        return {"status": "RECOMMENDED", "type": "single", **{k: best[k] for k in keys},
                "changes": [{"parameter": best["parameter"], "current_value": best["current_value"],
                             "recommended_value": best["recommended_value"]}], "disclaimer": DISCLAIMER}
    return {"status": "NO_EFFECTIVE_INTERVENTION_FOUND",
            "message": "No constrained single- or two-parameter change among the top root-cause candidates reduced the "
                       "model's predicted risk enough. This does not mean the sample is fine - only that the model sees no "
                       "safe lever among these features.", "disclaimer": DISCLAIMER}
