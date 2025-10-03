@echo off
REM Project Creation Helper Batch File
REM Author: Stephen Bilodeau
REM Date: August 2025

echo ==============================================
echo     Project Creation and Management Tool
echo ==============================================
echo.

:menu
echo Choose an option:
echo 1. Create new project (PowerShell)
echo 2. Create new project (Python)
echo 3. Generate project dashboard
echo 4. Open project organization guide
echo 5. Exit
echo.

set /p choice="Enter your choice (1-5): "

if "%choice%"=="1" goto powershell_create
if "%choice%"=="2" goto python_create
if "%choice%"=="3" goto generate_dashboard
if "%choice%"=="4" goto open_guide
if "%choice%"=="5" goto end

echo Invalid choice. Please try again.
echo.
goto menu

:powershell_create
echo.
echo Creating new project using PowerShell...
echo.

set /p name="Enter project name: "
set /p template="Enter template (default, python, web, research, data) [default]: "
set /p desc="Enter project description: "
set /p open_code="Open in VS Code when done? (y/n) [y]: "

if "%template%"=="" set template=default
if "%open_code%"=="" set open_code=y

if /i "%open_code%"=="y" (
    powershell -ExecutionPolicy Bypass -File "%~dp0new-project.ps1" -ProjectName "%name%" -Template "%template%" -Description "%desc%" -OpenVSCode
) else (
    powershell -ExecutionPolicy Bypass -File "%~dp0new-project.ps1" -ProjectName "%name%" -Template "%template%" -Description "%desc%"
)

echo.
echo Project creation complete!
echo.
pause
goto menu

:python_create
echo.
echo Creating new project using Python...
echo.

set /p name="Enter project name: "
set /p template="Enter template (default, python, web, research, data) [default]: "
set /p desc="Enter project description: "
set /p open_code="Open in VS Code when done? (y/n) [y]: "

if "%template%"=="" set template=default
if "%open_code%"=="" set open_code=y

if /i "%open_code%"=="y" (
    python "%~dp0new-project.py" "%name%" --template "%template%" --description "%desc%" --open-vscode
) else (
    python "%~dp0new-project.py" "%name%" --template "%template%" --description "%desc%"
)

echo.
echo Project creation complete!
echo.
pause
goto menu

:generate_dashboard
echo.
echo Generating project dashboard...
echo.

python "%~dp0project_dashboard.py"

echo.
echo Dashboard generation complete!
echo.
pause
goto menu

:open_guide
echo.
echo Opening project organization guide...
echo.

start "" "%~dp0..\PROJECT_ORGANIZATION_GUIDE.md"

echo.
pause
goto menu

:end
echo.
echo Thank you for using the Project Creation and Management Tool
echo.
exit /b
