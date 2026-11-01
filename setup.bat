@echo off
title Thiet lap moi truong - HaUI Student Prediction System
color 0A

echo ======================================================================
echo    KHOI TAO VA CAI DAT DU AN: HAUI STUDENT PREDICTION SYSTEM
echo ======================================================================
echo.

cd /d "%~dp0"

echo [1/5] Kiem tra cai dat Python tren he thong...
where python >nul 2>nul
if %ERRORLEVEL% NEQ 0 goto :no_python
python --version
if %ERRORLEVEL% NEQ 0 goto :no_python

echo.
echo [2/5] Kiem tra va khoi tao moi truong ao .venv...
if exist ".venv\Scripts\python.exe" goto :venv_exists
echo   - Dang tao moi truong ao .venv...
python -m venv .venv
if %ERRORLEVEL% NEQ 0 goto :venv_failed
echo   - Tao moi truong ao thanh cong.
goto :install_deps

:venv_exists
echo   - Moi truong ao .venv da ton tai san.

:install_deps
echo.
echo [3/5] Cai dat cac thu vien can thiet...
set "VENV_PYTHON=%~dp0.venv\Scripts\python.exe"

echo   - Dang nang cap pip...
"%VENV_PYTHON%" -m pip install --upgrade pip
if %ERRORLEVEL% NEQ 0 goto :install_failed

echo   - Cai dat du an va cac thu vien, bao gom XGBoost va CatBoost...
"%VENV_PYTHON%" -m pip install -e ".[dev]"
if %ERRORLEVEL% NEQ 0 goto :install_failed

echo   - Kiem tra XGBoost va CatBoost...
"%VENV_PYTHON%" -c "import xgboost, catboost; print('XGBoost:', xgboost.__version__); print('CatBoost:', catboost.__version__)"
if %ERRORLEVEL% NEQ 0 goto :install_failed

echo.
echo [4/5] Kiem tra file cau hinh moi truong .env...
if exist ".env" goto :env_exists
if exist ".env.example" (
    copy /y ".env.example" ".env" >nul
    echo   - Da tu dong tao file .env tu .env.example.
    goto :check_bootstrap
)
echo   - Khong tim thay file .env.example, bo qua.
goto :check_bootstrap

:env_exists
echo   - File .env da ton tai san.

:check_bootstrap
echo.
echo [5/5] Khoi tao thu muc va kiem tra bootstrap du an...
if exist "init_workspace.py" (
    "%VENV_PYTHON%" init_workspace.py
    if errorlevel 1 goto :install_failed
)
if exist "scripts\verify_bootstrap.py" (
    "%VENV_PYTHON%" scripts\verify_bootstrap.py
    if errorlevel 1 (
        echo   - Canh bao: Kiem tra khoi dong chua hoan hao.
    ) else (
        echo   - Kiem tra khoi dong UI va Model: THANH CONG [OK]
    )
)

echo.
echo ======================================================================
echo    THIET LAP HOAN TAT THANH CONG!
echo    Ban co the khoi dong ung dung bang cach chay file: run.bat
echo ======================================================================
echo.
pause
exit /b 0

:no_python
color 0C
echo [LOI] Khong tim thay Python tren he thong!
echo Vui long cai dat Python 3.10 tro len va tich chon "Add Python to PATH".
echo Tai tai: https://www.python.org/downloads/
echo.
pause
exit /b 1

:venv_failed
color 0C
echo [LOI] Tao moi truong ao .venv that bai!
echo.
pause
exit /b 1

:install_failed
color 0C
echo [LOI] Cai dat hoac kiem tra dependencies that bai!
echo Kiem tra thong bao loi ben tren, dac biet la pyproject.toml, XGBoost va CatBoost.
echo.
pause
exit /b 1
