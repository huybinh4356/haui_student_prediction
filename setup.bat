@echo off
title Thiet lap moi truong - HaUI Student Prediction System
color 0A

echo ======================================================================
echo    KHOI TAO VA CAI DAT DU AN: HAUI STUDENT PREDICTION SYSTEM
echo ======================================================================
echo.

cd /d "%~dp0"

:: 1. Kiem tra Python
echo [1/5] Kiem tra cai dat Python tren he thong...
where python >nul 2>nul
if %ERRORLEVEL% NEQ 0 goto :no_python
python --version

:: 2. Khoi tao moi truong ao (.venv)
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
:: 3. Cai dat thu vien
echo.
echo [3/5] Cai dat cac thu vien can thiet...
set "VENV_PYTHON=%~dp0.venv\Scripts\python.exe"
set "VENV_PIP=%~dp0.venv\Scripts\pip.exe"

echo   - Dang nang cap pip...
"%VENV_PYTHON%" -m pip install --upgrade pip --quiet

echo   - Cai dat goi du an va cac thu vien dependencies...
"%VENV_PIP%" install -e .[dev]
if %ERRORLEVEL% NEQ 0 goto :fallback_req
goto :after_install

:fallback_req
echo   - Co loi khi cai dat che do editable, thu cai dat qua requirements.txt...
"%VENV_PIP%" install -r requirements.txt
if %ERRORLEVEL% NEQ 0 goto :install_failed

:after_install
echo   - Cai dat thu vien hoan tat.

:: 4. Kiem tra file .env
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
:: 5. Khoi tao thu muc va kiem tra bootstrap
echo.
echo [5/5] Khoi tao thu muc va kiem tra bootstrap du an...
if exist "init_workspace.py" (
    "%VENV_PYTHON%" init_workspace.py
)
if exist "scripts\verify_bootstrap.py" (
    "%VENV_PYTHON%" scripts\verify_bootstrap.py >nul 2>nul
    if %ERRORLEVEL% EQU 0 (
        echo   - Kiem tra khoi dong UI va Model: THANH CONG [OK]
    ) else (
        echo   - Canh bao: Kiem tra khoi dong chua hoan hao, nhung van co the chay ung dung.
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
echo [LOI] Cai dat dependencies that bai! Vui long kiem tra lai ket noi mang.
echo.
pause
exit /b 1