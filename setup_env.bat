@echo off
setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    python -m venv .venv
    if errorlevel 1 exit /b 1
)

set "VENV_PYTHON=%~dp0.venv\Scripts\python.exe"

"%VENV_PYTHON%" -m pip install --upgrade pip
if errorlevel 1 exit /b 1

"%VENV_PYTHON%" -m pip install -e ".[dev]"
if errorlevel 1 exit /b 1

"%VENV_PYTHON%" -c "import xgboost; print('XGBoost version:', xgboost.__version__)"
if errorlevel 1 exit /b 1

echo Environment setup completed.
