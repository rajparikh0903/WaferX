import numpy as np
import pytest

from src.root_cause import ranking as rk
from src.root_cause.evidence_engine import analyze_anomaly


def _rc(synth, anomaly=None, k=5, row=0):
    x = synth["X"].iloc[synth["fail_rows"][row]]
    return synth["engine"].root_cause(x, anomaly=anomaly, top_k=k)["root_cause_candidates"]


def test_ranking_sorted_and_ranks_assigned(synth):
    c = _rc(synth)
    assert len(c) == 5
    assert [x["rank"] for x in c] == [1, 2, 3, 4, 5]
    assert all(c[i]["root_cause_score"] >= c[i + 1]["root_cause_score"] for i in range(4))


def test_scores_and_components_normalised(synth):
    for cand in _rc(synth, anomaly=analyze_anomaly(None, 0.9, True), k=10):
        assert 0 <= cand["root_cause_score"] <= 1
        assert all(v is None or 0 <= v <= 1 for v in cand["components"].values())
        assert abs(sum(cand["weights_used"].values()) - 1) < 1e-9


def test_missing_anomaly_score_still_works(synth):
    c = _rc(synth, anomaly=None)[0]
    assert c["components"]["anomaly"] is None and "anomaly" not in c["weights_used"]
    assert c["anomaly_evidence"] is None


def test_anomaly_score_used_when_provided(synth):
    c = _rc(synth, anomaly=analyze_anomaly(None, 0.9, True))[0]
    assert c["components"]["anomaly"] is not None and "anomaly" in c["weights_used"]


def test_true_cause_found_for_injected_failure(synth):
    causal = {synth["names"][i] for i in (2, 7, 13)}
    hits = sum(bool({c["feature"] for c in _rc(synth, k=3, row=r)} & causal) for r in range(20))
    assert hits >= 16


def test_combine_renormalises_weights():
    s, w = rk.combine({"shap": 1.0, "deviation": 0.0, "failure_association": None, "historical": None, "anomaly": None},
                      {"shap": 0.4, "deviation": 0.2, "failure_association": 0.15, "historical": 0.15, "anomaly": 0.1})
    assert s == pytest.approx(0.4 / 0.6) and sum(w.values()) == pytest.approx(1)


@pytest.mark.parametrize("kw", [{"anomaly_score": 1.5}, {"anomaly_score": -0.1}, {"anomaly_score": "x"},
                                {"anomaly_label": "false"}, {"anomaly_score": float("nan")}])
def test_anomaly_validation(kw):
    with pytest.raises(ValueError):
        analyze_anomaly(None, **kw)


def test_anomaly_label_only_maps_to_score():
    assert analyze_anomaly(1, anomaly_label=True)["effective_score"] == 1.0
    assert analyze_anomaly(1)["available"] is False


def test_similarity_excludes_self_and_labels(synth):
    sim = synth["bundle"].similarity
    row = 5
    out = sim.similar(synth["bundle"].ref.Z[row], sim.risk[row], k=5, exclude_id=int(sim.ids[row]))
    assert len(out) == 5 and all(m["sample_id"] != int(sim.ids[row]) for m in out)
    assert all(m["label"] in ("PASS", "FAIL") and 0 <= m["similarity"] <= 1 for m in out)
