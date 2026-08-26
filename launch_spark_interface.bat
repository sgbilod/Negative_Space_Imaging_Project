@echo off
REM Launch Spark Interface
REM Negative Space Imaging Project
REM Copyright (c) 2025 Stephen Bilodeau. All rights reserved.

echo ==============================================
echo Negative Space Imaging - Spark Interface
echo ==============================================

REM Set Spark configuration
if not defined SPARK_DRIVER_MEMORY set SPARK_DRIVER_MEMORY=4g
if not defined SPARK_EXECUTOR_MEMORY set SPARK_EXECUTOR_MEMORY=4g
if not defined SPARK_EXECUTOR_CORES set SPARK_EXECUTOR_CORES=4

REM Activate virtual environment
if exist ".venv\Scripts\activate.bat" (
    call .venv\Scripts\activate.bat
) else if exist ".venv-hpc\Scripts\activate.bat" (
    call .venv-hpc\Scripts\activate.bat
)

REM Check for Spark
where spark-submit >nul 2>&1
if errorlevel 1 (
    echo Warning: Apache Spark not found
    echo Install Spark or set SPARK_HOME environment variable
    exit /b 1
)

echo Starting Spark interface...
echo Driver Memory: %SPARK_DRIVER_MEMORY%
echo Executor Memory: %SPARK_EXECUTOR_MEMORY%

if exist "spark_interface\main.py" (
    spark-submit ^
        --driver-memory %SPARK_DRIVER_MEMORY% ^
        --executor-memory %SPARK_EXECUTOR_MEMORY% ^
        --executor-cores %SPARK_EXECUTOR_CORES% ^
        spark_interface\main.py %*
) else (
    echo Spark interface module not found
    python -c "from spark_interface import start; start()"
)
