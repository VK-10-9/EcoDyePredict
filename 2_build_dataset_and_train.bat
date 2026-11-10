@echo off
REM Builds datasets and trains models for EcoDyePredict
REM Default: trains all dyes found in data\raw\
REM Specific dye:  2_build_dataset_and_train.bat "methylene blue"
cd /d %~dp0
if exist venv\Scripts\activate.bat (
    call venv\Scripts\activate.bat
)
if "%~1"=="" (
    echo Training models for all dyes in data\raw...
    python src\train_all_dyes.py
) else (
    echo Training model for dye: %~1
    python src\preprocess.py --dye "%~1"
    if errorlevel 1 ( pause & exit /b 1 )
    python src\train.py
)
pause
