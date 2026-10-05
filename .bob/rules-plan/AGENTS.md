# AGENTS.md

This file provides guidance to agents when working with code in this repository.

## Non-Obvious Architectural Constraints

- **No external database** — all persistence is file-based: Parquet/CSV for data, joblib for model artifacts, JSON for metadata. Everything under `src/final_integrated_backend/models/` and `data/`.
- **Root-cause scoring is a weighted sum of 5 evidence sources** with weights defined in `config.yaml` under `root_cause.weights` (shap: 0.4, deviation: 0.2, failure_association: 0.15, historical: 0.15, anomaly: 0.1). Changing the algorithm requires updating these weights.
- **`RootCausePipeline` is a lazy singleton** — initialized on first HTTP request, not at startup. Health checks via `GET /health` do NOT load the pipeline.
- **Two model stacks coexist**: LightGBM (production SECOM model) and XGBoost (used in conftest synthetic tests). They share the same `AnalysisEngine` interface.
- **Wafer image intelligence requires separate model artifacts** under `models/image_intelligence/` (6 `.pt`/`.joblib` files for wm+mix datasets). The `GET /health` endpoint reports `"missing_artifacts"` if they are absent — image endpoints will 503 until provided.
- **Frontend ↔ Backend coupling**: the frontend calls `POST /analyze-with-image` (multipart form with `sample_id` or `features_json`) for unified analysis. The JSON-only endpoints (`/analyze`, `/predict`, etc.) are used by the MCP server and direct API consumers.
- **MCP server is stateless** — each tool call opens a new `httpx.AsyncClient` with a 120s timeout. The server itself holds no pipeline state.
- **`AnalysisEngine` is intentionally dataset-agnostic** to allow the same code path in synthetic validation tests (no SECOM artifacts needed) and production (SECOM artifacts).
