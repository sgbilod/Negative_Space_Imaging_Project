@echo off
REM Project Creation Helper Batch File Launcher
REM Author: Stephen Bilodeau
REM Date: August 2025

REM Path to the actual project management scripts
SET SCRIPTS_DIR=C:\Users\sgbil\OneDrive\Desktop\Negative_Space_Imaging_Project\scripts

REM Run the project generator
cd /d "%SCRIPTS_DIR%" && python project_generator.py gui

REM Exit with the same code as the project generator
exit /b %errorlevel%
