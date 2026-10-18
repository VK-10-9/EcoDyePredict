@echo off
REM Tests the whole pipeline + app on SYNTHETIC data. Not for the report.
cd /d %~dp0
call venv\Scripts\activate
python src\make_demo_data.py
python src\preprocess.py --raw data\demo --out data\demo\dataset.csv
python src\train.py --data data\demo\dataset.csv --out models_demo --demo
set ECODYE_MODELS=models_demo
set ECODYE_DATA=data/demo/dataset.csv
streamlit run app.py
