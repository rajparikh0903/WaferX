"""FastAPI service for YieldTwin process + wafer-image intelligence.

Run:
    python -m uvicorn src.inference.api:app --reload --port 8000
"""
from __future__ import annotations

import json
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, model_validator

from src.image_intelligence import image_model_service
from src.inference.pipeline import RootCausePipeline
from src.utils import load_config

app = FastAPI(title="YieldTwin Root-Cause + Wafer Image Intelligence", version="2.0.0")
_cfg = load_config()
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cfg["api"]["cors_origins"],
    allow_methods=["*"],
    allow_headers=["*"],
)
_pipe: Optional[RootCausePipeline] = None


def pipe() -> RootCausePipeline:
    global _pipe
    if _pipe is None:
        try:
            _pipe = RootCausePipeline()
        except FileNotFoundError as e:
            raise HTTPException(
                503,
                f"Model artefacts not found - run `python run_pipeline.py train` ({e})",
            )
    return _pipe


class SampleRef(BaseModel):
    sample_id: Optional[int] = Field(None, ge=0, description="Row id in the SECOM file (0-based)")
    features: Optional[dict[str, Optional[float]]] = Field(
        None,
        description='e.g. {"Sensor 60": 12.3}; omitted sensors are treated as missing',
    )

    @model_validator(mode="after")
    def _one(self):
        if (self.sample_id is None) == (self.features is None):
            raise ValueError("provide exactly one of sample_id or features")
        return self


class AnalyzeRequest(SampleRef):
    anomaly_score: Optional[float] = Field(None, ge=0, le=1)
    anomaly_label: Optional[bool] = None
    top_k: Optional[int] = Field(None, ge=1, le=20)


class WhatIfRequest(SampleRef):
    feature: str
    values: Optional[list[float]] = Field(None, min_length=1, max_length=200)


def _call(fn, *a, **k):
    try:
        return fn(*a, **k)
    except KeyError as e:
        raise HTTPException(404, str(e).strip("'\""))
    except ValueError as e:
        raise HTTPException(422, str(e))


def _parse_features_json(features_json: str | None) -> dict[str, float | None] | None:
    if features_json is None or not features_json.strip():
        return None
    try:
        value = json.loads(features_json)
    except json.JSONDecodeError as e:
        raise HTTPException(422, f"features_json is not valid JSON: {e.msg}")
    if not isinstance(value, dict):
        raise HTTPException(422, "features_json must decode to an object")
    return value


def _select_sample_ref(sample_id: int | None, features_json: str | None) -> tuple[int | None, dict | None]:
    features = _parse_features_json(features_json)
    if (sample_id is None) == (features is None):
        raise HTTPException(422, "provide exactly one of sample_id or features_json")
    return sample_id, features


@app.post("/predict")
def predict(r: SampleRef):
    return _call(pipe().predict, sample_id=r.sample_id, features=r.features)


@app.post("/root-cause")
def root_cause(r: AnalyzeRequest):
    return _call(
        pipe().root_cause,
        sample_id=r.sample_id,
        features=r.features,
        anomaly_score=r.anomaly_score,
        anomaly_label=r.anomaly_label,
        top_k=r.top_k,
    )


@app.post("/what-if")
def what_if(r: WhatIfRequest):
    return _call(
        pipe().what_if,
        r.feature,
        sample_id=r.sample_id,
        features=r.features,
        values=r.values,
    )


@app.post("/analyze")
def analyze(r: AnalyzeRequest):
    return _call(
        pipe().analyze,
        sample_id=r.sample_id,
        features=r.features,
        anomaly_score=r.anomaly_score,
        anomaly_label=r.anomaly_label,
        top_k=r.top_k,
    )


@app.post("/anomaly")
def anomaly(r: SampleRef):
    return _call(pipe().anomaly, sample_id=r.sample_id, features=r.features)


@app.post("/image/analyze")
async def image_analyze(
    wafer_map: UploadFile = File(..., description="Wafer map PNG/JPG/WebP/NPY/NPZ"),
    dataset: str = Form("wm", description="wm for wafer-map pattern labels; mix for binary-code classes"),
):
    """Run the exported DINO+ViT image classifier and image anomaly detector."""
    if dataset not in {"wm", "mix"}:
        raise HTTPException(422, "dataset must be 'wm' or 'mix'")
    try:
        data = await wafer_map.read()
        if not data:
            raise ValueError("uploaded wafer map is empty")
        return image_model_service.predict_bytes(data, wafer_map.filename, dataset=dataset)
    except (ValueError, FileNotFoundError) as e:
        raise HTTPException(422, str(e))
    except RuntimeError as e:
        raise HTTPException(503, str(e))


@app.get("/image/model-info")
def image_model_info(dataset: str = "wm"):
    if dataset not in {"wm", "mix"}:
        raise HTTPException(422, "dataset must be 'wm' or 'mix'")
    try:
        return image_model_service.info(dataset)
    except (ValueError, FileNotFoundError) as e:
        raise HTTPException(422, str(e))
    except RuntimeError as e:
        raise HTTPException(503, str(e))


@app.post("/analyze-with-image")
async def analyze_with_image(
    wafer_map: UploadFile = File(..., description="Wafer map PNG/JPG/WebP/NPY/NPZ"),
    dataset: str = Form("wm"),
    sample_id: int | None = Form(None),
    features_json: str | None = Form(None),
    anomaly_score: float | None = Form(None),
    anomaly_label: bool | None = Form(None),
    top_k: int | None = Form(None),
):
    """Frontend-friendly unified endpoint: process analysis + wafer-image intelligence."""
    if dataset not in {"wm", "mix"}:
        raise HTTPException(422, "dataset must be 'wm' or 'mix'")
    selected_sample_id, features = _select_sample_ref(sample_id, features_json)
    if anomaly_score is not None and not 0 <= anomaly_score <= 1:
        raise HTTPException(422, "anomaly_score must be between 0 and 1")
    if top_k is not None and not 1 <= top_k <= 20:
        raise HTTPException(422, "top_k must be between 1 and 20")

    try:
        image_data = await wafer_map.read()
        if not image_data:
            raise ValueError("uploaded wafer map is empty")

        process_result = pipe().analyze(
            sample_id=selected_sample_id,
            features=features,
            anomaly_score=anomaly_score,
            anomaly_label=anomaly_label,
            top_k=top_k,
        )
        image_result = image_model_service.predict_bytes(
            image_data, wafer_map.filename, dataset=dataset
        )
        return {
            "process_intelligence": process_result,
            "wafer_image_intelligence": image_result,
            "integration": {
                "process_model": "SECOM LightGBM + SHAP + Isolation Forest + root-cause engine",
                "image_model": f"{dataset.upper()} DINOv2 + ViT embeddings + MLP + image Isolation Forest",
            },
        }
    except (KeyError,) as e:
        raise HTTPException(404, str(e).strip("'\""))
    except ValueError as e:
        raise HTTPException(422, str(e))
    except FileNotFoundError as e:
        raise HTTPException(503, str(e))
    except RuntimeError as e:
        raise HTTPException(503, str(e))


@app.get("/anomaly-info")
def anomaly_info():
    return pipe().anomaly_detector.info()


@app.get("/health")
def health():
    image_artifacts = all(
        p.exists()
        for p in (
            image_model_service.folder / "classifier_mlp_wm_full.pt",
            image_model_service.folder / "classifier_mlp_wm_full_prep.joblib",
            image_model_service.folder / "detector_wm_full_dino+vit.joblib",
            image_model_service.folder / "classifier_mlp_mix_full.pt",
            image_model_service.folder / "classifier_mlp_mix_full_prep.joblib",
            image_model_service.folder / "detector_mix_full_dino+vit.joblib",
        )
    )
    return {
        "status": "ok",
        "root_cause": "ready",
        "anomaly_detection": "ready",
        "image_intelligence": "configured" if image_artifacts else "missing_artifacts",
        "image_datasets": ["wm", "mix"] if image_artifacts else [],
    }


@app.get("/model-info")
def model_info():
    return pipe().model_info()


@app.get("/feature-importance")
def feature_importance(top_n: int = 20):
    return pipe().feature_importance(max(1, min(top_n, 100)))
