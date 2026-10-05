import io

import numpy as np
from fastapi.testclient import TestClient

from src.inference import api


class FakeImageService:
    folder = type("Folder", (), {})()

    def predict_bytes(self, data, filename=None, dataset="wm"):
        return {
            "dataset": dataset,
            "label": "Center",
            "confidence": 0.95,
            "top3": {"Center": 0.95, "none": 0.04, "Donut": 0.01},
            "anomaly_score_classifier": 0.96,
            "anomaly_label": True,
            "detector_score": 0.42,
            "input": {"filename": filename, "map_shape": [52, 52]},
        }

    def info(self, dataset="wm"):
        return {"dataset": dataset, "classes": ["Center", "none"]}



def test_image_route(monkeypatch):
    client = TestClient(api.app)
    monkeypatch.setattr(api, "image_model_service", FakeImageService())
    r = client.post(
        "/image/analyze",
        files={"wafer_map": ("wafer.png", b"fake", "image/png")},
        data={"dataset": "wm"},
    )
    assert r.status_code == 200
    assert r.json()["label"] == "Center"


def test_analyze_with_image_route(monkeypatch):
    client = TestClient(api.app)
    monkeypatch.setattr(api, "image_model_service", FakeImageService())
    monkeypatch.setattr(
        api,
        "pipe",
        lambda: type(
            "FakePipe",
            (),
            {"analyze": lambda self, **kwargs: {"prediction": {"predicted_class": "FAIL"}}},
        )(),
    )
    r = client.post(
        "/analyze-with-image",
        files={"wafer_map": ("wafer.png", b"fake", "image/png")},
        data={"dataset": "wm", "sample_id": "1363"},
    )
    assert r.status_code == 200
    body = r.json()
    assert "process_intelligence" in body
    assert "wafer_image_intelligence" in body
    assert body["wafer_image_intelligence"]["label"] == "Center"
