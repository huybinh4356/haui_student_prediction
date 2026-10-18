@echo off
title HaUI Student Prediction System
color 0B

echo ===================================================
echo KHOI DONG UNG DUNG HAUI STUDENT PREDICTION
echo ===================================================
echo.

cd /d "%~dp0"

if not exist .venv\Scripts\activate.bat (
    color 0C
    echo [LOI] Khong tim thay moi truong ao!
    echo Vui long nhay dup vao file "setup.bat" de cai dat truoc.
    pause
    exit /b 1
)

echo Dang kich hoat moi truong ao...
call .venv\Scripts\activate.bat

echo Dang mo giao dien nguoi dung...
python main.py

if %ERRORLEVEL% NEQ 0 (
    color 0C
    echo.
    echo [LOI] Ung dung da dung hoac co loi xay ra (Ma loi: %ERRORLEVEL%).
    pause
)