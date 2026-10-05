from src.inference.pipeline import RootCausePipeline


def test_anomaly_is_automatic():
    pipe = RootCausePipeline()
    result = pipe.analyze(sample_id=23, top_k=5)
    assert result["anomaly"]["available"] is True
    assert result["anomaly"]["model_type"] == "unsupervised_anomaly_detection"
    assert 0.0 <= result["anomaly"]["anomaly_score"] <= 1.0
    assert isinstance(result["anomaly"]["anomaly_label"], bool)
    assert result["root_cause_candidates"]


def test_existing_root_cause_still_works():
    pipe = RootCausePipeline()
    result = pipe.analyze(sample_id=11, top_k=5)
    assert result["prediction"]["failure_probability"] >= 0.0
    assert result["root_cause_candidates"]
    assert result["historical_matches"] is not None
    assert result["counterfactuals"] is not None
