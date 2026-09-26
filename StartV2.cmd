@echo off
cd /d "%~dp0"
set "V2_PYTHON=.venv\Scripts\python.exe"
if not exist "%V2_PYTHON%" set "V2_PYTHON=..\..\work\venv\Scripts\python.exe"
if not exist "%V2_PYTHON%" (
  echo Python environment missing. Run Install.ps1 first.
  pause
  exit /b 1
)
"%V2_PYTHON%" -m osumapper.v2 ui --home "local-data-v2" --port 7861
if errorlevel 1 pause
