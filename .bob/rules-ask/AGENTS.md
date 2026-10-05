# AGENTS.md

This file provides guidance to agents when working with code in this repository.

## Non-Obvious Documentation Context

- **`src/final_integrated_backend/` is the real backend** — `src/final_integrated_backend/src/` is the Python package root (not `src/` at the repo root).
- **`FINAL_README.md` and `FINAL_MODEL_CARD.md`** in `src/final_integrated_backend/` contain authoritative implementation notes; the top-level `README.md` is the unfilled hackathon template.
- **The anomaly model is Isolation Forest** (trained on PASS samples only); `models/secom_rf/` contains a RandomForestClassifier for PASS/FAIL that is **not** an anomaly detector — the naming is misleading.
- **SECOM has no ground-truth anomaly labels** — FAIL labels are used only as an evaluation proxy, not as anomaly training targets.
- **`run_pipeline.py train`** actually calls `src/training/train_xgboost.py` despite the production model being LightGBM; the LightGBM trainer is at `src/training/train_final_lightgbm.py` but is invoked separately.
- **`src/bob_mcp/`** is the IBM Bob MCP integration (FastMCP), not general backend code.
- **Frontend env var**: only one — `NEXT_PUBLIC_API_URL` pointing to the FastAPI backend. No auth or secrets needed for the frontend.
- **`config.yaml` is the single source of truth** for all model hyperparameters, thresholds, paths, and algorithm weights — not hardcoded in Python files.
