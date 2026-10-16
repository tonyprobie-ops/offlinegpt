@echo off
setlocal enabledelayedexpansion
chcp 65001 >nul
cd /d "%~dp0"
title Offline AI Studio
set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"
set "VENV=%~dp0.venv"
set "PY=%VENV%\Scripts\python.exe"

echo ============================================
echo  Offline AI Studio - local, free, no cloud
echo ============================================
echo.

if not exist "%PY%" (
  echo [1/4] Creating the virtual environment in .venv ...
  where py >nul 2>nul
  if !errorlevel! equ 0 (
    py -3 -m venv "%VENV%"
  ) else (
    python -m venv "%VENV%"
  )
  if not exist "%PY%" (
    echo.
    echo ERROR: Python 3.10 or newer was not found.
    echo Install it from https://www.python.org/downloads/windows/ and tick "Add python.exe to PATH".
    pause
    exit /b 1
  )
) else (
  echo [1/4] Reusing the existing virtual environment.
)

echo [2/4] Updating pip ...
"%PY%" -m pip install --upgrade pip --quiet --disable-pip-version-check

echo [3/4] Installing requirements ...
"%PY%" -m pip install -r requirements.txt --quiet --disable-pip-version-check
if errorlevel 1 (
  echo.
  echo ERROR: Could not install the requirements. Check your internet connection for this first run only.
  pause
  exit /b 1
)

echo [4/4] Starting Offline AI Studio ...
echo       The app starts the LM Studio server and loads the model by itself.
echo       Nothing is sent anywhere - you can unplug the network once it is running.
echo.
"%PY%" main.py
set "CODE=%errorlevel%"
if not "%CODE%"=="0" (
  echo.
  echo The app exited with code %CODE%.
  pause
)
endlocal
exit /b %CODE%
