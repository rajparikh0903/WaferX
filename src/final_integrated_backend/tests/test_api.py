import pytest
from fastapi.testclient import TestClient

from src.inference import api


@pytest.fixture(scope="module")
def client(real_pipeline):
    api._pipe = real_pipeline
    return TestClient(api.app)


def test_analyze_valid_sample(client):
    r = client.post("/analyze", json={"sample_id": 1363, "anomaly_score": 0.91, "anomaly_label": True})
    assert r.status_code == 200
    j = r.json()
    for k in ["sample", "prediction", "anomaly", "root_cause_candidates", "historical_matches", "counterfactuals",
              "recommendation", "disclaimer"]:
        assert k in j
    assert len(j["root_cause_candidates"]) == 5 and j["anomaly"]["available"] is True
    assert "not a physically validated" in j["disclaimer"]


def test_analyze_without_anomaly(client):
    j = client.post("/analyze", json={"sample_id": 1363}).json()
    assert j["anomaly"]["available"] is True


def test_predict_and_root_cause_and_what_if(client):
    assert client.post("/predict", json={"sample_id": 1363}).status_code == 200
    rc = client.post("/root-cause", json={"sample_id": 1363}).json()
    feat = rc["root_cause_candidates"][0]["feature"]
    w = client.post("/what-if", json={"sample_id": 1363, "feature": feat})
    assert w.status_code == 200 and len(w.json()["curve"]) > 1


def test_features_payload(client):
    r = client.post("/predict", json={"features": {"Sensor 60": 2.1}})
    assert r.status_code == 200 and r.json()["sample"]["split"] == "external"


@pytest.mark.parametrize("payload,code", [
    ({"sample_id": 999999}, 404),
    ({"sample_id": "abc"}, 422),
    ({"sample_id": -1}, 422),
    ({}, 422),
    ({"sample_id": 1, "features": {"Sensor 1": 1.0}}, 422),
    ({"sample_id": 1, "anomaly_score": 1.7}, 422),
    ({"features": {"Not a sensor": 1.0}}, 422),
    ({"features": {"Sensor 1": "text"}}, 422),
])
def test_invalid_and_malformed_inputs(client, payload, code):
    assert client.post("/analyze", json=payload).status_code == code


def test_what_if_unknown_feature(client):
    assert client.post("/what-if", json={"sample_id": 1363, "feature": "Sensor 9999"}).status_code == 422


def test_info_endpoints(client):
    m = client.get("/model-info").json()
    assert m["model"] == "lightgbm" and "threshold" in m
    fi = client.get("/feature-importance?top_n=5").json()
    assert len(fi["features"]) == 5
