# YieldTwin — Final Root Cause + Anomaly Detection Backend

## What changed

The existing root-cause engine was kept intact. The missing anomaly layer is now automatic and runs inside `/analyze`.

### Actual anomaly model
- Isolation Forest
- trained only on PASS rows from the TRAIN split
- 200 estimators
- max_samples = 0.70
- max_features = 1.0
- anomaly score = empirical percentile relative to TRAIN-PASS anomaly-score distribution
- alert threshold = 0.90

SECOM does **not** contain ground-truth anomaly labels. FAIL is never used as the anomaly training target. FAIL is used only as an evaluation proxy in `models/anomaly/metrics.json`.

## Important artifact clarification

`models/secom_rf/secom_rf_k100_exact.joblib` was inspected and tested separately. It is a **RandomForestClassifier** for PASS/FAIL classification, not an anomaly detector. The supplied config identifies it as `SECOM RF k=100` with ANOVA top-100 feature selection and a 200-tree Random Forest.

It should not be mislabeled as an anomaly model. The final stack therefore uses the supplied RF only as an optional secondary classifier artifact and uses Isolation Forest for true unsupervised anomaly detection.

## API

Existing endpoints remain:
- `POST /predict`
- `POST /root-cause`
- `POST /what-if`
- `POST /analyze`
- `GET /model-info`
- `GET /feature-importance`

New anomaly endpoints:
- `POST /anomaly`
- `GET /anomaly-info`
- `GET /health`

`POST /analyze` now automatically includes a real `anomaly` block unless the caller explicitly supplies `anomaly_score` / `anomaly_label` as an override.

## Run

```powershell
cd C:\Users\rajpa\Desktop\IBM\Final\final_root_cause_engine
python -m uvicorn src.inference.api:app --reload --port 8000
```

Then:

```text
http://127.0.0.1:8000/docs
```

## Quick checks

```powershell
python -m scripts.check_stack
```

or directly:

```powershell
$body = @{ sample_id = 23; top_k = 5 } | ConvertTo-Json
Invoke-RestMethod `
  -Uri http://127.0.0.1:8000/analyze `
  -Method POST `
  -ContentType "application/json" `
  -Body $body | ConvertTo-Json -Depth 30
```

Sample 23 is a useful anomaly demo: in the included test evaluation it produced an anomaly score of about 0.991 and crossed the 0.90 alert threshold.

Sample 11 remains a useful root-cause demo: its anomaly score is about 0.617 (not anomalous at the configured alert threshold), while the root-cause engine still produces live SHAP/statistical/historical evidence.

## Wafer Image Intelligence Integration

The backend now includes the exported wafer-image models from `wafer_export` under `models/image_intelligence/`.

New endpoints:

- `POST /image/analyze` — classify a wafer map and return visual anomaly scores.
- `GET /image/model-info?dataset=wm|mix` — inspect the selected image model.
- `POST /analyze-with-image` — recommended frontend endpoint when both process/sensor data and a wafer image are available.

See `IMAGE_INTELLIGENCE.md` for the exact request fields and response structure.

### Recommended frontend flow

For a process-only screen, keep using `POST /analyze`.

For a screen that has a wafer image too, send one multipart request to `POST /analyze-with-image`:

- `wafer_map`: uploaded wafer image / `.npy` / `.npz`
- `dataset`: `wm` or `mix`
- `sample_id`: a SECOM row id **or** `features_json`: the JSON sensor-feature object
- optional `anomaly_score`, `anomaly_label`, `top_k`

The response separates `process_intelligence` and `wafer_image_intelligence`, making it straightforward for the frontend to render both panels without changing the existing backend contracts.
