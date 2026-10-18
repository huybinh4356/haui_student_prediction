@echo off
setlocal
title Khoi dong - HaUI Student Prediction System
color 0B

echo ======================================================================
echo    KHOI DONG UNG DUNG HAUI STUDENT PREDICTION SYSTEM
echo ======================================================================
echo.

cd /d "%~dp0"

set "VENV_PYTHON=%~dp0.venv\Scripts\python.exe"

:: Kiem tra su ton tai cua moi truong ao
if not exist "%VENV_PYTHON%" (
    color 0C
    echo [LOI] Khong tim thay moi truong ao .venv!
    echo.
    echo Vui long chay file "setup.bat" truoc de tu dong thiet lap va cai dat thu vien.
    echo ======================================================================
    echo.
    pause
    exit /b 1
)

:: Kiem tra file .env
if not exist ".env" (
    if exist ".env.example" (
        copy /y ".env.example" ".env" >nul
        echo [THONG BAO] Da tu dong tao file .env tu .env.example.
    )
)

echo Dang khoi tao va mo giao dien PyQt6...
echo.

"%VENV_PYTHON%" main.py

if errorlevel 1 (
    color 0C
    echo.
    echo ======================================================================
    echo [LOI] Ung dung ket thuc voi ma loi hoac co su co xay ra.
    echo Vui long kiem tra lai thong bao tren hoac chay "setup.bat" de dong bo moi truong.
    echo ======================================================================
    echo.
    pause
    exit /b 1
)

exit /b 0