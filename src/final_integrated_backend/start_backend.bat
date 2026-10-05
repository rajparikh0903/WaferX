@echo off
setlocal
cd /d %~dp0
python -m uvicorn src.inference.api:app --host 0.0.0.0 --port 8000
endlocal
