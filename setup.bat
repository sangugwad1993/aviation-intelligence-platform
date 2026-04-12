@echo off
REM ============================================================================
REM Aviation Intelligence Platform — One-Click Setup (Windows)
REM ============================================================================
REM Usage:
REM   setup.bat              Full native setup (Python venv + deps + tests)
REM   setup.bat --docker     Docker-only setup (build image + launch container)
REM ============================================================================

setlocal enabledelayedexpansion

echo.
echo =============================================
echo   Aviation Intelligence Platform — Setup
echo =============================================
echo.

if "%1"=="--docker" goto :docker_setup
if "%1"=="--help" goto :show_help
if "%1"=="-h" goto :show_help

REM --- Check Python --------------------------------------------------------
echo [INFO]  Checking Python...
where python >nul 2>&1
if %errorlevel% neq 0 (
    echo [WARN]  Python not found.
    echo.
    echo   Please install Python 3.10+ from: https://www.python.org/downloads/
    echo   IMPORTANT: Check "Add Python to PATH" during installation.
    echo.
    pause
    exit /b 1
)

python --version 2>&1 | findstr /R "3\.1[0-9]" >nul
if %errorlevel% neq 0 (
    python --version 2>&1 | findstr /R "3\.[2-9][0-9]" >nul
    if %errorlevel% neq 0 (
        echo [FAIL]  Python 3.10+ required.
        python --version
        pause
        exit /b 1
    )
)

for /f "tokens=*" %%i in ('python --version 2^>^&1') do echo [OK]    %%i

REM --- Create virtual environment ------------------------------------------
echo.
echo [INFO]  Creating virtual environment...
if not exist ".venv" (
    python -m venv .venv
    echo [OK]    Virtual environment created at .venv\
) else (
    echo [OK]    Virtual environment already exists.
)

REM --- Activate venv -------------------------------------------------------
call .venv\Scripts\activate.bat
echo [OK]    Activated .venv

REM --- Upgrade pip ---------------------------------------------------------
echo [INFO]  Upgrading pip...
python -m pip install --upgrade pip -q
echo [OK]    pip upgraded

REM --- Install dependencies ------------------------------------------------
echo.
echo [INFO]  Installing dependencies (this may take 2-3 minutes)...
pip install -r requirements-dev.txt -q
echo [OK]    All Python dependencies installed

REM --- Create directories --------------------------------------------------
if not exist "data" mkdir data
if not exist "artifacts" mkdir artifacts
echo [OK]    Data directories ready

REM --- Run tests -----------------------------------------------------------
echo.
echo [INFO]  Running test suite...
echo.
set PYTHONPATH=%cd%
python -m pytest tests/ -v --no-cov --tb=short 2>&1 | more +0
if %errorlevel% equ 0 (
    echo.
    echo [OK]    All tests passed!
) else (
    echo.
    echo [WARN]  Some tests failed — the dashboard will still work.
)

goto :setup_done

REM --- Docker setup --------------------------------------------------------
:docker_setup
echo [INFO]  Checking Docker...
where docker >nul 2>&1
if %errorlevel% neq 0 (
    echo [FAIL]  Docker not found.
    echo.
    echo   Please install Docker Desktop from: https://www.docker.com/products/docker-desktop/
    echo.
    pause
    exit /b 1
)

docker info >nul 2>&1
if %errorlevel% neq 0 (
    echo [WARN]  Docker is not running. Please start Docker Desktop first.
    pause
    exit /b 1
)

echo [OK]    Docker is running.
echo.
echo [INFO]  Building Docker image...
docker compose build
echo [OK]    Docker image built

echo [INFO]  Starting dashboard container...
docker compose up -d
echo [OK]    Container running

goto :setup_done

REM --- Help ----------------------------------------------------------------
:show_help
echo.
echo Aviation Intelligence Platform — Setup Script (Windows)
echo.
echo Usage:
echo   setup.bat              Full native setup (Python venv + deps + tests)
echo   setup.bat --docker     Docker-only setup (build image + launch)
echo.
echo Prerequisites:
echo   Native:  Python 3.10+ (https://python.org)
echo   Docker:  Docker Desktop (https://docker.com)
echo.
echo Login credentials:
echo   admin  / aviation2024     (full access + admin panel)
echo   viewer / readonly2024     (flight map + predictions)
echo.
exit /b 0

REM --- Done ----------------------------------------------------------------
:setup_done
echo.
echo =============================================
echo   Setup Complete!
echo =============================================
echo.
echo   To start the dashboard:
echo.
echo     .venv\Scripts\activate
echo     set PYTHONPATH=%cd%
echo     streamlit run dashboard\app.py
echo.
echo   Or use Docker:
echo.
echo     docker compose up --build -d
echo.
echo   Login credentials:
echo     admin  / aviation2024     (full access + admin panel)
echo     viewer / readonly2024     (flight map + predictions)
echo.
echo   Dashboard URL: http://localhost:8501
echo.

set /p LAUNCH="  Launch dashboard now? [Y/n] "
if /i "%LAUNCH%"=="" goto :launch
if /i "%LAUNCH%"=="y" goto :launch
if /i "%LAUNCH%"=="Y" goto :launch
goto :eof

:launch
echo [INFO]  Starting Streamlit dashboard...
call .venv\Scripts\activate.bat
set PYTHONPATH=%cd%
streamlit run dashboard\app.py --server.port=8501 --server.address=localhost

:eof
endlocal
