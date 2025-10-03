@echo off
REM Setup Database Integration Environment for Negative Space Imaging Project
REM Windows Batch File

echo ===================================
echo Database Integration Setup Utility
echo ===================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo Python is not installed or not in PATH.
    echo Please install Python 3.7 or higher.
    exit /b 1
)

echo Python is installed, proceeding with setup...
echo.

REM Parse command-line arguments
set ALL=
set INSTALL_DEPS=
set CREATE_DIRS=
set INIT_DB=
set TEST=
set FORCE=
set CONFIG="deployment/config/database.yaml"

:parse_args
if "%~1"=="" goto end_parse_args
if /i "%~1"=="--all" set ALL=--all
if /i "%~1"=="--install-deps" set INSTALL_DEPS=--install-deps
if /i "%~1"=="--create-dirs" set CREATE_DIRS=--create-dirs
if /i "%~1"=="--init-db" set INIT_DB=--init-db
if /i "%~1"=="--test" set TEST=--test
if /i "%~1"=="--force" set FORCE=--force
if /i "%~1"=="--config" set CONFIG="%~2" && shift
shift
goto parse_args
:end_parse_args

REM Build the command
set CMD=python deployment/setup_database.py

if defined ALL set CMD=%CMD% --all
if defined INSTALL_DEPS set CMD=%CMD% --install-deps
if defined CREATE_DIRS set CMD=%CMD% --create-dirs
if defined INIT_DB set CMD=%CMD% --init-db
if defined TEST set CMD=%CMD% --test
if defined FORCE set CMD=%CMD% --force
set CMD=%CMD% --config %CONFIG%

echo Running setup with command:
echo %CMD%
echo.

REM Execute the setup script
%CMD%

if %errorlevel% neq 0 (
    echo.
    echo Setup failed. Please check the logs for details.
    exit /b 1
) else (
    echo.
    echo Setup completed successfully.
)

echo.
echo To manage the database, use the following commands:
echo.
echo python deployment/database_deploy.py --deploy    : Deploy the database
echo python deployment/database_deploy.py --verify    : Verify the database
echo python deployment/database_deploy.py --migrate   : Run migrations
echo python deployment/database_deploy.py --backup    : Backup the database
echo python deployment/database_deploy.py --restore   : Restore from backup
echo.
echo python deployment/test_database_deployment.py    : Test the database
echo.

pause
