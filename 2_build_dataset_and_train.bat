@echo off
REM Merges data\raw\*.csv, locks one dye, trains models, saves models\best_model.joblib
REM To force a dye:  2_build_dataset_and_train.bat "methylene blue"
cd /d %~dp0
call venv\Scripts\activate
if "%~1"=="" ( python src\preprocess.py ) else ( python src\preprocess.py --dye %1 )
if errorlevel 1 ( pause & exit /b 1 )
python src\train.py
pause
