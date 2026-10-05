import json

import numpy as np
import pytest

from src.inference.pipeline import risk_level


def test_model_loads_and_predicts_probabilities(real_pipeline):
    P = real_pipeline
    pr = P.model.predict_proba(P.X_model.iloc[:50])[:, 1]
    assert pr.shape == (50,) and np.all((pr >= 0) & (pr <= 1))
    assert P.features == P.meta["feature_list"]
    assert 0 < P.meta["threshold"] < 1


def test_prediction_output_contract(real_pipeline):
    out = real_pipeline.predict(sample_id=1363)["prediction"]
    assert set(["sample_id", "predicted_class", "failure_probability", "risk_level"]) <= set(out)
    assert out["predicted_class"] in ("PASS", "FAIL") and out["risk_level"] in ("LOW", "MEDIUM", "HIGH", "CRITICAL")
    assert out["predicted_class"] == ("FAIL" if out["failure_probability"] >= out["decision_threshold"] else "PASS")


def test_risk_levels_monotonic(cfg):
    order = ["LOW", "MEDIUM", "HIGH", "CRITICAL"]
    lv = [order.index(risk_level(p, 0.2, cfg)) for p in np.linspace(0, 1, 101)]
    assert lv == sorted(lv) and set(lv) == {0, 1, 2, 3}


def test_shap_feature_count_and_additivity(synth):
    X = synth["X"].iloc[synth["te"][:20]]
    sv = synth["shap"].shap_values(X)
    assert sv.shape == (20, len(synth["names"]))
    logit = synth["model"].predict(X, output_margin=True)
    assert np.allclose(sv.sum(1) + synth["shap"].base_value, logit, atol=1e-3)


def test_local_explanation_signs(synth):
    x = synth["X"].iloc[synth["fail_rows"][0]]
    row = synth["shap"].shap_values(x.to_frame().T)[0]
    loc = synth["shap"].local_explanation(x, row, 0.5)
    assert all(t["contribution"] > 0 for t in loc["top_positive"])
    assert all(t["contribution"] < 0 for t in loc["top_negative"])


def test_global_importance_sorted(synth):
    gi = synth["shap"].global_importance(synth["X"].iloc[:300])
    assert gi["mean_abs_shap"].is_monotonic_decreasing and len(gi) == len(synth["names"])
