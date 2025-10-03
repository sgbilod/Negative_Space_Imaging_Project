@echo off
REM Start backend server
start cmd /k "cd api && python -m uvicorn api:app --reload --port 8000"

REM Start frontend development server
start cmd /k "cd frontend && npm start"
