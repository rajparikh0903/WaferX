import numpy as np
import pytest

from src.counterfactual.constraints import intervention_scale
from src.counterfactual.generator import candidate_values, predict_with_values


def _setup(synth, feat_idx=2, row=0):
    x = synth["X"].iloc[synth["fail_rows"][row]]
    f = synth["names"][feat_idx]
    return x, f, synth["bundle"].constraints.loc[f]


def test_candidates_respect_range_max_and_min_change(synth, cfg):
    cf = cfg["counterfactual"]
    for r in range(10):
        x, f, c = _setup(synth, row=r)
        v = float(x[f])
        vals = candidate_values(v, c, cf)
        scale = intervention_scale(v, c["span"], cf["scale_span_floor"])
        if len(vals) == 0:
            continue
        assert vals.min() >= c["op_lo"] - 1e-9 and vals.max() <= c["op_hi"] + 1e-9
        assert np.all(np.abs(vals - v) <= cf["max_change_pct"] * scale + 1e-9)
        assert np.all(np.abs(vals - v) >= cf["min_change_pct"] * scale - 1e-9)
        assert vals.min() >= c["obs_min"] and vals.max() <= c["obs_max"]


def test_no_candidates_when_value_far_outside_operating_range(synth, cfg):
    x, f, c = _setup(synth)
    far = float(c["op_hi"] + 10 * c["span"])
    assert len(candidate_values(far, c, cfg["counterfactual"])) == 0


def test_risk_comes_from_model_inference(synth):
    x, f, c = _setup(synth)
    vals = np.array([float(x[f]) - 1.0, float(x[f]) + 1.0])
    got = predict_with_values(synth["model"], x, synth["names"], {f: vals})
    for v, g in zip(vals, got):
        x2 = x.copy(); x2[f] = v
        assert g == pytest.approx(synth["engine"].proba(x2), abs=1e-6)


def test_recommendation_is_constrained_and_improves_risk(synth, cfg):
    cf = cfg["counterfactual"]
    found = 0
    for r in range(15):
        x = synth["X"].iloc[synth["fail_rows"][r]]
        a = synth["engine"].analyze(x)
        rec = a["recommendation"]
        assert a["disclaimer"].startswith("This is a model-based counterfactual recommendation")
        if rec["status"] != "RECOMMENDED":
            continue
        found += 1
        assert rec["risk_reduction"] >= cf["min_abs_reduction"] - 1e-9
        assert rec["predicted_risk"] < rec["current_risk"]
        assert rec["intervention_magnitude"] <= 2 * cf["max_change_pct"] + 1e-9   # pair = sum of two <=10%
        for ch in rec["changes"]:
            c = synth["bundle"].constraints.loc[ch["parameter"]]
            assert c["op_lo"] - 1e-9 <= ch["recommended_value"] <= c["op_hi"] + 1e-9
            assert ch["recommended_value"] != ch["current_value"]
    assert found > 0


def test_objective_formula_and_balanced_choice(synth, cfg):
    cf = cfg["counterfactual"]
    for r in range(15):
        x = synth["X"].iloc[synth["fail_rows"][r]]
        for cfx in synth["engine"].analyze(x)["counterfactuals"]:
            if cfx["status"] != "RECOMMENDED":
                continue
            ok = [p for p in cfx["curve"] if p["risk_reduction"] >= cf["min_abs_reduction"]]
            for p in cfx["curve"]:
                assert p["objective"] == pytest.approx(p["risk"] + cf["lambda_penalty"] * p["intervention_magnitude"])
            assert cfx["objective"] == pytest.approx(min(p["objective"] for p in ok))
            return
    pytest.skip("no recommendation found")


def test_what_if_flags_out_of_constraint_values(synth):
    x, f, c = _setup(synth)
    out = synth["engine"].what_if(x, f, values=[float(x[f]), float(c["op_hi"]) + 5 * float(c["span"])])
    assert [p["within_constraints"] for p in out["curve"]] == [True, False] or \
           [p["within_constraints"] for p in out["curve"]] == [False, True]
    assert all(0 <= p["risk"] <= 1 for p in out["curve"])
    with pytest.raises(ValueError):
        synth["engine"].what_if(x, "not a feature")
