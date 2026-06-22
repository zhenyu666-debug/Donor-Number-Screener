@echo off
REM start.bat - launch the donor-screener-pbp FastAPI service.
setlocal
set ROOT=%~dp0
cd /d %ROOT%
echo [pbp] starting FastAPI on :8001 ...
python -m uvicorn src.29_pbp_api:app --host 127.0.0.1 --port 8001
endlocal
