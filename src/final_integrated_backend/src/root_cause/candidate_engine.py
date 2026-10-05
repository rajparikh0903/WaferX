"""Root-Cause Candidate Engine: SHAP + deviation + failure association + historical + external anomaly.

Outputs HYPOTHESES. Nothing here proves causality, and SECOM sensors are anonymous - no physical meaning is assumed.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.root_cause import ranking as rk


class RootCauseEngine:
    def __init__(self, ref, similarity, cfg: dict):
        self.ref, self.sim, self.cfg = ref, similarity, cfg
        self.rc = cfg["root_cause"]

    def rank(self, x_row: pd.Series, shap_row: np.ndarray, model_features: list[str], proba: float,
             anomaly: dict | None = None, top_k: int | None = None, exclude_id=None) -> list[dict]:
        rc, ref = self.rc, self.ref
        top_k = top_k or rc["top_k"]
        shap = pd.Series(shap_row, index=model_features)[ref.sensor_cols]
        shap_rank = shap.rank(ascending=False, method="first").astype(int)
        pos_max = float(max(shap.max(), 0.0))
        pool = shap.sort_values(ascending=False).head(rc["candidate_pool"]).index.tolist()
        z = ref.robust_z(x_row)
        z_by = dict(zip(ref.sensor_cols, z))

        nbrs, hist_abn = [], None
        if self.sim is not None:
            nbrs = self.sim.similar(z, proba, k=rc["hist_k"], exclude_id=exclude_id, only_fail=True)
            if nbrs:
                Zn = ref.Z[[n["row"] for n in nbrs]]
                hist_abn = (np.abs(Zn) >= rc["z_abnormal"]) & (np.sign(Zn) == np.sign(z)[None, :])
        a_score = None if not anomaly or not anomaly.get("available") else anomaly["effective_score"]

        cands = []
        for f in pool:
            j = ref.sensor_cols.index(f)
            zj, v = float(z_by[f]), float(x_row[f])
            c_shap = float(max(shap[f], 0.0) / pos_max) if pos_max > 0 else 0.0
            c_dev = float(min(abs(zj) / rc["z_cap"], 1.0))
            a = ref.assoc.loc[f]
            c_assoc = float(a["effect_pct"])
            c_hist = None
            n_abn = None
            if hist_abn is not None:
                n_abn = int(hist_abn[:, j].sum())
                c_hist = float(n_abn / len(nbrs)) if abs(zj) >= rc["hist_min_current_z"] else 0.0
            c_anom = None if a_score is None else float(a_score * c_dev)
            comps = {"shap": c_shap, "deviation": c_dev, "failure_association": c_assoc,
                     "historical": c_hist, "anomaly": c_anom}
            score, eff_w = rk.combine(comps, rc["weights"])
            cmp = ref.compare(f, v)
            cands.append({
                "feature": f, "current_value": v,
                "normal_stats": cmp["pass"], "failure_stats": cmp["fail"],
                "shap_contribution": float(shap[f]), "shap_rank": int(shap_rank[f]),
                "deviation_sigma": zj,
                "deviation_percentile_in_pass": cmp["percentile_in_pass"],
                "failure_association": {"effect_size_r": float(a["effect_r"]), "auc": float(a["auc"]),
                                        "p_value": float(a["p_value"]), "q_value": float(a["q_value"]),
                                        "significant": bool(a["significant"])},
                "historical_evidence": None if c_hist is None else {
                    "similar_fail_samples_checked": len(nbrs), "abnormal_same_direction": n_abn},
                "anomaly_evidence": None if c_anom is None else {
                    "anomaly_score": a_score, "combined_with_deviation": c_anom},
                "components": comps,
                "component_levels": {k: rk.level(v_, rc) for k, v_ in comps.items()},
                "weights_used": eff_w,
                "root_cause_score": score,
                "confidence": rk.confidence(score, comps, rc),
                "evidence_strength": rk.evidence_strength(score, rc),
                "comparison": cmp,
                "label": "Root-Cause Candidate (hypothesis, not proven cause)",
            })
        top = rk.rank_candidates(cands, top_k)
        for c in top:
            c["explanation"] = self._explain(c)
        return top

    @staticmethod
    def _explain(c: dict) -> str:
        f, cmp = c["feature"], c["comparison"]
        parts = [f"{f} adds {c['shap_contribution']:+.3f} log-odds toward FAIL (SHAP rank #{c['shap_rank']})."]
        pb = cmp["pass"]
        if pb:
            parts.append(f"Current value {c['current_value']:.6g} is {c['deviation_sigma']:+.1f}σ (robust) from the PASS "
                         f"median {pb['median']:.6g} (percentile {cmp['percentile_in_pass']:.0f} of PASS samples).")
        a = c["failure_association"]
        parts.append("PASS and FAIL distributions differ significantly (FDR q=%.3g, effect r=%+.2f)." % (a["q_value"], a["effect_size_r"])
                     if a["significant"] else
                     "PASS vs FAIL separation for this feature is not statistically significant after FDR correction.")
        if cmp["closer_to"] and cmp["fail"]:
            parts.append(f"Value is closer to the FAIL median ({cmp['fail']['median']:.6g})." if cmp["closer_to"] == "FAIL"
                         else f"Value is closer to the PASS median than to the FAIL median ({cmp['fail']['median']:.6g}).")
        h = c["historical_evidence"]
        if h and (c["components"]["historical"] or 0) > 0:
            parts.append(f"{h['abnormal_same_direction']}/{h['similar_fail_samples_checked']} of the most similar historical "
                         "FAIL samples deviate in the same direction.")
        elif h:
            parts.append("Historical evidence is not counted because this sample's own deviation is below the minimum.")
        if c["anomaly_evidence"]:
            parts.append(f"External anomaly score {c['anomaly_evidence']['anomaly_score']:.2f} supports this deviation.")
        parts.append("Hypothesis ranked by model and statistical evidence; not a proven root cause.")
        return " ".join(parts)
