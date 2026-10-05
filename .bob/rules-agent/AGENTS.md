# AGENTS.md

This file provides guidance to agents when working with code in this repository.

## Non-Obvious Coding Rules

- **Always use `to_py()` from `src/final_integrated_backend/src/utils.py`** before serializing any pipeline output — it handles `numpy` types, `NaN`, `inf`, `pd.NaT`, and nested structures. Missing this causes `json.JSONDecodeError` at the FastAPI response layer.
- **`load_config()` must be called from within `src/final_integrated_backend/`** (or with an explicit path). It resolves paths relative to the config file location and injects a `_root` key — callers that skip this will get `KeyError: '_root'` when calling `resolve()`.
- **`AnalysisEngine` and `RootCausePipeline` are separate classes**: `AnalysisEngine` is dataset-agnostic and used in tests with synthetic data; `RootCausePipeline` loads actual SECOM artifacts. Do not conflate them.
- **Backend imports use `src.*` package paths** (e.g., `from src.inference.pipeline import ...`). This only works when the working directory is `src/final_integrated_backend/`. Launching uvicorn or pytest from the repo root breaks all imports.
- **MCP tools in `src/bob_mcp/server.py`** forward to the deployed Render URL by default. Set `YIELDTWIN_API_URL` to redirect to a local backend.
- **Feature names are always `"Sensor {i}"` strings**, zero-based integer indices from the SECOM dataset. Hardcoded in `config.yaml` as `feature_name_template`. Don't use raw integer column indices anywhere.
- **No linter config exists** (no `.flake8`, `ruff.toml`, `pyproject.toml`). Follow the existing style: `from __future__ import annotations`, type hints as `X | None` (not `Optional[X]`), 4-space indentation.
- **Frontend has no TypeScript strict config beyond defaults** — `tsconfig.json` uses Next.js defaults. No custom path aliases.
