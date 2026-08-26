@echo off
REM Negative Space Imaging System Launcher
REM Copyright (c) 2025 Stephen Bilodeau. All rights reserved.

echo Starting Negative Space Imaging System...

REM Activate virtual environment
call .venv\Scripts\activate.bat

REM Run system setup if needed
python scripts\setup_system.py

REM Launch the system
python scripts\launch_system.py

pause
