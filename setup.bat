@echo off
title Setup HaUI Student Prediction System
color 0A

echo ===================================================
echo KHOI TAO MOI TRUONG DU AN HAUI STUDENT PREDICTION
echo ===================================================
echo.

cd /d "%~dp0"

echo [1/4] Dang kiem tra va tao moi truong ao (.venv)...
if not exist .venv (
    python -m venv .venv
    echo =^> Da tao moi truong ao thanh cong.
) else (
    echo =^> Moi truong ao da ton tai.
)

echo.
echo [2/4] Kich hoat moi truong ao...
call .venv\Scripts\activate.bat

echo.
echo [3/4] Cap nhat pip va cai dat thu vien (pip install -e .[dev])...
python -m pip install --upgrade pip
pip install -e .[dev]

echo.
echo [4/4] Kiem tra file cau hinh (.env)...
if not exist .env (
    if exist .env.example (
        copy .env.example .env
        echo =^> Da tao file .env tu .env.example.
    ) else (
        echo =^> Khong tim thay .env.example.
    )
) else (
    echo =^> File .env da ton tai.
)

echo.
echo ===================================================
echo HOAN TAT THIET LAP!
echo Ban co the chay file "run.bat" de khoi dong ung dung.
echo ===================================================
pause