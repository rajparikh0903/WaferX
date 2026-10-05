"""Transparent Root-Cause Candidate Score.

    score = sum_i( w_i * c_i ) / sum_i( w_i )          i over the evidence that is AVAILABLE

c_i are all normalised to [0, 1]; weights live in config.yaml (root_cause.weights).
  shap                 = max(shap_j, 0) / max_k max(shap_k, 0)
  deviation            = min(|robust z_j| / z_cap, 1)
  failure_association  = percentile rank of |rank-biserial effect| among all sensors
  historical           = share of similar historical FAIL samples abnormal in the same direction
                         (0 unless the sample itself deviates >= hist_min_current_z)
  anomaly              = external anomaly score x deviation   (only when an anomaly score is supplied)
"""
from __future__ import annotations

COMPONENTS = ["shap", "deviation", "failure_association", "historical", "anomaly"]


def combine(components: dict, weights: dict) -> tuple[float, dict]:
    """Return (score, effective_weights). Missing (None) components are dropped and weights re-normalised."""
    avail = {k: weights[k] for k in COMPONENTS if components.get(k) is not None and weights.get(k, 0) > 0}
    tot = sum(avail.values())
    if tot <= 0:
        return 0.0, {}
    eff = {k: w / tot for k, w in avail.items()}
    return float(sum(eff[k] * components[k] for k in eff)), eff


def level(x: float | None, rc_cfg: dict) -> str | None:
    if x is None:
        return None
    return "HIGH" if x >= rc_cfg["strength_high"] else "MEDIUM" if x >= rc_cfg["strength_medium"] else "LOW"


def confidence(score: float, components: dict, rc_cfg: dict) -> str:
    c = rc_cfg["confidence"]
    n_strong = sum(1 for v in components.values() if v is not None and v >= c["strong_component"])
    if score >= c["high_score"] and n_strong >= c["high_min_strong"]:
        return "HIGH"
    if score >= c["medium_score"] and n_strong >= c["medium_min_strong"]:
        return "MEDIUM"
    return "LOW"


def evidence_strength(score: float, rc_cfg: dict) -> str:
    return "Strong" if score >= rc_cfg["evidence_strong"] else "Medium" if score >= rc_cfg["evidence_medium"] else "Weak"


def rank_candidates(cands: list[dict], top_k: int) -> list[dict]:
    cands = sorted(cands, key=lambda c: (-c["root_cause_score"], -c["shap_contribution"]))[:top_k]
    for i, c in enumerate(cands, 1):
        c["rank"] = i
    return cands
