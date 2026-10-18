@echo off
REM Reads every PDF in data\papers and writes data\raw\psh_rag.csv
cd /d %~dp0
call venv\Scripts\activate
python src\extract_rag.py
pause
