"""Historical similarity: cosine k-NN on process fingerprints (top-SHAP robust z-scores + out-of-fold model risk).

Supporting evidence only - a similar historical sample is NOT causal proof.
"""
from __future__ import annotations

import numpy as np
from sklearn.neighbors import NearestNeighbors


class SimilarityIndex:
    def __init__(self, ref, fingerprint_features: list[str], risk, ids, labels, timestamps, sim_cfg: dict,
                 threshold: float, anomaly_scores=None):
        self.fp_idx = np.array([ref.sensor_cols.index(f) for f in fingerprint_features])
        self.fp_names = list(fingerprint_features)
        self.zc = float(sim_cfg["z_clip"])
        self.rw = float(sim_cfg["risk_weight"])
        self.risk_scale = max(2.0 * float(threshold), 1e-6)
        self.Z = ref.Z
        self.ids = np.asarray(ids)
        self.labels = np.asarray(labels).astype(int)
        self.risk = np.asarray(risk, dtype=float)
        self.ts = None if timestamps is None else [str(t) for t in timestamps]
        self.anomaly = None if anomaly_scores is None else np.asarray(anomaly_scores, dtype=float)
        F = np.vstack([self.fingerprint(self.Z[i], self.risk[i]) for i in range(len(self.ids))])
        self.F = F
        self.nn_all = NearestNeighbors(metric="cosine", algorithm="brute").fit(F)
        self.fail_rows = np.where(self.labels == 1)[0]
        self.nn_fail = NearestNeighbors(metric="cosine", algorithm="brute").fit(F[self.fail_rows]) if len(self.fail_rows) else None

    def fingerprint(self, z_full, risk: float) -> np.ndarray:
        z = np.clip(np.asarray(z_full, dtype=float)[self.fp_idx], -self.zc, self.zc) / self.zc
        r = self.rw * min(max(float(risk), 0.0) / self.risk_scale, 1.0)
        return np.concatenate([z, [r]])

    def similar(self, z_full, risk: float, k: int = 5, exclude_id=None, only_fail: bool = False) -> list[dict]:
        q = self.fingerprint(z_full, risk)
        if not np.linalg.norm(q) > 0:
            return []
        nn, rows_map = (self.nn_fail, self.fail_rows) if only_fail else (self.nn_all, np.arange(len(self.ids)))
        if nn is None:
            return []
        n_req = min(k + 1, len(rows_map))
        dist, idx = nn.kneighbors(q.reshape(1, -1), n_neighbors=n_req)
        out = []
        for d, i in zip(dist[0], idx[0]):
            row = int(rows_map[i])
            if exclude_id is not None and int(self.ids[row]) == int(exclude_id):
                continue
            sim = float(max(0.0, 1.0 - d))
            out.append({"sample_id": int(self.ids[row]), "similarity": sim, "similarity_pct": round(100 * sim, 1),
                        "label": "FAIL" if self.labels[row] == 1 else "PASS",
                        "model_risk_oof": float(self.risk[row]),
                        "timestamp": None if self.ts is None else self.ts[row], "row": row})
            if len(out) == k:
                break
        return out
