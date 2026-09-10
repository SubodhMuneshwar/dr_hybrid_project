@echo off
REM Activate virtual environment
call .venv\Scripts\activate

REM Run Flask app directly using Python
python app/app.py

pause
