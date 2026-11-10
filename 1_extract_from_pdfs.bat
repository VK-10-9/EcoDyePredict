@echo off
REM Reads every PDF in data\papers and writes data\raw\psh_rag.csv using Gemini AI
cd /d %~dp0
if exist venv\Scripts\activate.bat (
    call venv\Scripts\activate.bat
)
python src\extract_rag.py
pause
