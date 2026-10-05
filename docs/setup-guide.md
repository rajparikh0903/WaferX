# Setup Guide

> **This file is read by the automated evaluation pipeline. Be precise and complete.**

## Prerequisites

Before you begin, ensure you have the following installed:

- [ ] Python 3.11.x
- [ ] Node.js 18+
- [ ] npm
- [ ] Git
- [ ] Internet access for the first DINOv2/ViT backbone download if image inference is used

## Environment Variables

The current local prototype does not require `.env.example`, an external database, Docker, watsonx.ai credentials, or Slack credentials.

Create `src/frontend/.env.local` with:

```env
NEXT_PUBLIC_API_URL=http://127.0.0.1:8000
```

The project-level IBM Bob MCP configuration is stored at `.bob/mcp.json`. For local use, it should point to:

```text
YIELDTWIN_API_URL=http://127.0.0.1:8000
```

No API keys or external IBM credentials are required for the current local MCP integration.

| Variable | Description | Required |
|---|---|---|
| `NEXT_PUBLIC_API_URL` | Local FastAPI backend URL used by the frontend | Yes |
| `YIELDTWIN_API_URL` | Local YieldTwin API URL used by the IBM Bob MCP server in `.bob/mcp.json` | For IBM Bob/MCP |

## Installation

```powershell
# 1. Clone the repository
git clone https://github.com/rajparikh0903/WaferX.git
cd WaferX

# 2. Install backend dependencies
cd src/final_integrated_backend
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt

# 3. Preload image backbones (once, before first wafer-image inference)
python scripts/preload_image_backbones.py
```

For the frontend, open a second terminal from the repository root:

```powershell
# 4. Install frontend dependencies
cd src/frontend
npm install
```

Create `src/frontend/.env.local` containing:

```env
NEXT_PUBLIC_API_URL=http://127.0.0.1:8000
```

If PowerShell prevents virtual environment activation, allow scripts for this terminal session and activate again:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

### IBM Bob MCP dependencies

From the repository root:

```powershell
pip install -r src/bob_mcp/requirements.txt
```

Open the project in IBM Bob and ensure the `yieldtwin` MCP server from `.bob/mcp.json` is enabled.

## Running the Application

```powershell
# Start the backend in Terminal 1, from src/final_integrated_backend
.\.venv\Scripts\Activate.ps1
python -m uvicorn src.inference.api:app --reload --host 127.0.0.1 --port 8000
```

Backend: `http://127.0.0.1:8000`  
Swagger UI: `http://127.0.0.1:8000/docs`  
Health check: `http://127.0.0.1:8000/health`

```powershell
# Start the frontend in Terminal 2, from src/frontend
npm run dev
```

The frontend will be available at: `http://localhost:3000`

### IBM Bob

Keep the backend running on:

```text
http://127.0.0.1:8000
```

Then open the repository in IBM Bob. The `yieldtwin` MCP server in `.bob/mcp.json` connects Bob to the local YieldTwin APIs.

Example Bob test:

```text
Use the YieldTwin MCP tools to investigate sample 100.
Return the predicted class, failure probability, anomaly status,
and top root-cause candidates.
```

## Running Tests

Backend tests, from `src/final_integrated_backend` with the virtual environment active:

```powershell
pytest -q
```

Frontend production build, from `src/frontend`:

```powershell
npm run build
```

## Quick Demo (Optional)

1. Start the backend on port `8000`.
2. Start the frontend on port `3000`.
3. Open `http://localhost:3000/investigate` and use sample ID `100` to run process investigation.
4. Open `http://localhost:3000/what-if` and test a sensor counterfactual.
5. Open `http://localhost:3000/image-analysis` and upload a supported wafer image (`.png`, `.jpg`, `.jpeg`, `.webp`, `.npy`, or `.npz`).
6. Open the AI Engineer Assistant and ask it to investigate sample `100`.

## Troubleshooting

| Issue | Solution |
|---|---|
| `ModuleNotFoundError: No module named 'src'` | Run backend commands from `src/final_integrated_backend`, activate its `.venv`, and use the documented `python -m uvicorn` command. |
| `ModuleNotFoundError` for Python packages | Activate the backend `.venv` and run `python -m pip install -r requirements.txt` from `src/final_integrated_backend`. |
| PowerShell blocks virtual environment activation | Run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` in that terminal, then activate `.venv` again. |
| Image model cannot load | Run `python scripts/preload_image_backbones.py` from `src/final_integrated_backend` with internet access. |
| Frontend cannot reach backend | Check that FastAPI is running at `127.0.0.1:8000` and `NEXT_PUBLIC_API_URL` in `src/frontend/.env.local` matches it. |
| Frontend API still uses an old URL | Stop and restart `npm run dev` after changing `.env.local`. |
| TypeScript build fails | Run `npm run build` in `src/frontend` and fix the reported file/line. |
| IBM Bob shows YieldTwin MCP as disconnected | Check `.bob/mcp.json`, confirm the Python interpreter can import `mcp`, and refresh/restart the MCP server in Bob. |
| Swagger UI does not load | Check `http://127.0.0.1:8000/health` first, then open `http://127.0.0.1:8000/docs`. |
