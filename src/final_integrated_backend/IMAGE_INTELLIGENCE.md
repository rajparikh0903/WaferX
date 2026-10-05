# YieldTwin Image Intelligence Integration

The backend now contains the exported wafer image models under:

`models/image_intelligence/`

The image inference stack is:

**Wafer map → DINOv2 + ViT embeddings → trained MLP classifier → wafer-pattern label/confidence**

and in parallel:

**DINOv2 + ViT embeddings → PCA → Isolation Forest → visual anomaly score**

## Supported model families

- `wm`: 9 wafer-pattern classes (`Center`, `Donut`, `Edge-Loc`, `Edge-Ring`, `Loc`, `Near-full`, `Random`, `Scratch`, `none`)
- `mix`: 38 binary-code classes from the exported model

## API

### `POST /image/analyze`

Multipart form-data:

- `wafer_map`: image file (`PNG/JPG/WebP`) or `.npy` / `.npz`
- `dataset`: `wm` or `mix` (default `wm`)

The response includes:

- predicted image label
- confidence
- top-3 classes
- normal-class probability
- classifier anomaly score (`1 - normal_probability`)
- classifier anomaly label (`score >= 0.5`)
- raw image Isolation Forest detector score

### `POST /analyze-with-image`

This is the recommended endpoint for the frontend when it has both process data and a wafer image.

Multipart form-data:

- `wafer_map`: required image/map file
- `dataset`: `wm` or `mix`
- exactly one of `sample_id` or `features_json`
- `anomaly_score`: optional process-anomaly override
- `anomaly_label`: optional process-anomaly override
- `top_k`: optional root-cause candidate count

The response contains two clearly separated blocks:

- `process_intelligence`: existing SECOM prediction, anomaly detection, SHAP explanation, root-cause candidates, historical matches, counterfactuals and recommendation
- `wafer_image_intelligence`: wafer defect pattern classification and image anomaly results

Existing JSON endpoints such as `/analyze`, `/predict`, `/root-cause`, `/what-if`, and `/anomaly` remain available and unchanged.

## Important deployment note

The exported `.pt` classifier files contain the trained MLP head, not the large pretrained DINOv2/ViT backbone weights. `timm` therefore loads those pretrained backbones at runtime. On the deployment machine, run the backend once with internet access (or otherwise pre-cache the timm weights) before using `/image/analyze`.

This is a model packaging limitation of the supplied `wafer_export` artifact, not a change to the trained classifier.
