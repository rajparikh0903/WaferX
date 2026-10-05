@echo off
setlocal
cd /d %~dp0
if not exist node_modules (
  echo Installing frontend dependencies...
  npm install
)
npm run dev
endlocal
