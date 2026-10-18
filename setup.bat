@echo off
cd /d %~dp0
if not exist venv ( python -m venv venv )
call venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
echo.
set /p X="Also install the paper-extraction packages (large, ~2 GB with torch)? [y/N] "
if /i "%X%"=="y" pip install -r requirements-extract.txt
if not exist .env copy .env.example .env
echo.
echo Setup done. Edit .env and paste your ANTHROPIC_API_KEY if you will run the extraction step.
pause
