# AGENTS.md

This file provides guidance to agents when working with code in this repository.

## Project Overview

WaferX / YieldTwin: semiconductor defect investigation tool. Three components:
- **Backend** (`src/final_integrated_backend/`) — FastAPI + LightGBM + SHAP + Isolation Forest
- **Frontend** (`src/frontend/`) — Next.js 14 + React + Tailwind + Recharts
- **MCP server** (`src/bob_mcp/`) — FastMCP bridge for IBM Bob

## Commands

### Backend (run from `src/final_integrated_backend/`)
```bash
# Start API server
python -m uvicorn src.inference.api:app --reload --port 8000
# or
python run_pipeline.py serve --port 8000

# Train model (required before running tests against real pipeline)
python run_pipeline.py train

# Run all tests
pytest tests/ -v

# Run a single test file
pytest tests/test_api.py -v

# Run a single test
pytest tests/test_api.py::test_analyze_valid_sample -v
```

### Frontend (run from `src/frontend/`)
```bash
npm run dev      # dev server on port 3000
npm run build    # production build
```

### MCP server (run from `src/bob_mcp/`)
```bash
pip install -r requirements.txt
python server.py
```

## Critical Architecture Notes

- **All backend tests must be run from `src/final_integrated_backend/`** — `conftest.py` adds that directory to `sys.path` using `Path(__file__).resolve().parents[1]`. Running pytest from the repo root will fail with import errors.
- **Model artifacts must exist before API or `real_pipeline` tests run.** The `real_pipeline` fixture calls `pytest.skip()` if `models/metadata/metadata.json` is missing. Run `python run_pipeline.py train` first.
- **No `pyproject.toml` or `setup.cfg`** — there is no project-level pytest config. All test discovery relies on pytest defaults.
- **`to_py()` in `src/utils.py` is the mandatory serializer** for all API responses — it converts numpy/pandas types and maps `NaN`/`inf` → `None`. Never use `json.dumps` directly on pipeline output.
- **All config lives in `config.yaml`** (not environment variables). Loaded via `load_config()` from `src/utils.py`; injects `_root` key with absolute path of the config file's parent. Path resolution always goes through `resolve(cfg, key)`.
- **`RootCausePipeline` is a lazy singleton** in `api.py` — initialized on first request, raises HTTP 503 if artifacts are missing.
- **Frontend API URL** is controlled by `NEXT_PUBLIC_API_URL` env var (default: `http://localhost:8000`). Set in `src/frontend/.env` (copy from `.env.example`).
- **`YIELDTWIN_API_URL`** env var controls which backend the MCP server hits (default: `https://waferx.onrender.com`).

## Code Style

- All Python files use `from __future__ import annotations` at the top.
- Type hints use the modern union syntax (`dict[str, Any] | None`) due to the future annotations import.
- Feature names throughout the codebase follow the template `"Sensor {i}"` (from `config.yaml`).
- Sensor/feature columns are always `str` keys, even when derived from numeric indices.
- Tests use `scope="session"` fixtures for expensive objects (model training, pipeline loading).
- The `synth` conftest fixture trains a fast XGBoost model (not LightGBM) for unit tests — do not assume the test model matches production.

## Known Limitations (per submission.yaml)
- Anomaly detection panel and defect-image classification have placeholder/mock functionality.
- Sample list uses demo sample IDs only.
- What-if counterfactuals require engineering validation before any physical process change.
