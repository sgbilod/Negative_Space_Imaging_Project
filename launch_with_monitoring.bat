@echo off
REM Launch with Monitoring Enabled
REM Negative Space Imaging Project
REM Copyright (c) 2025 Stephen Bilodeau. All rights reserved.

echo ==============================================
echo Negative Space Imaging - Launch with Monitoring
echo ==============================================

REM Configuration
if not defined PROMETHEUS_PORT set PROMETHEUS_PORT=9090
if not defined GRAFANA_PORT set GRAFANA_PORT=3000
if not defined APP_PORT set APP_PORT=8080

REM Activate virtual environment
if exist ".venv\Scripts\activate.bat" (
    call .venv\Scripts\activate.bat
)

REM Start monitoring stack if Docker is available
where docker-compose >nul 2>&1
if not errorlevel 1 (
    echo Starting monitoring stack...
    docker-compose -f docker-compose.performance.yml up -d prometheus grafana 2>nul
)

REM Set monitoring environment variables
set ENABLE_METRICS=true
set METRICS_PORT=9100
set PROMETHEUS_MULTIPROC_DIR=%TEMP%\prometheus_multiproc
if not exist "%PROMETHEUS_MULTIPROC_DIR%" mkdir "%PROMETHEUS_MULTIPROC_DIR%"

echo.
echo Monitoring Configuration:
echo   Prometheus: http://localhost:%PROMETHEUS_PORT%
echo   Grafana: http://localhost:%GRAFANA_PORT%
echo   Metrics: http://localhost:%METRICS_PORT%/metrics
echo.

REM Launch main application
if exist "cli.py" (
    python cli.py %*
) else if exist "api\server.py" (
    python api\server.py %*
) else (
    echo Starting demo mode...
    python pipeline_demo.py --mode full %*
)
