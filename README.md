# EcoDyePredict

Streamlit app predicting dye removal (%) by pumpkin seed husk from pH, dosage, concentration, time and temperature.

## Run order (Windows, Python 3.11)
1. **`setup.bat`** - creates the venv, installs packages, creates `.env`.
2. Put paper PDFs in `data/papers/` (name = DOI with `/` -> `_`, see the README there). Paste your `ANTHROPIC_API_KEY` in `.env`.
3. **`1_extract_from_pdfs.bat`** - RAG extraction -> `data/raw/psh_rag.csv`.
   Manual rows go in `data/raw/psh_<name>.csv` (copy `TEMPLATE_psh_yourname.csv`). Both are merged automatically.
4. **Verify rows against the papers.** Fix or delete bad rows in the CSVs. Rows tagged `graph_estimate` are excluded by default.
5. **`2_build_dataset_and_train.bat`** - merge, validate, lock ONE dye, train 4 models, save `models/best_model.joblib`.
   Force a dye: `2_build_dataset_and_train.bat "methylene blue"`.
6. **`3_run_app.bat`** - opens the app.

**Test everything first, with no data:** `0_demo_smoke_test.bat` (synthetic data, clearly bannered, never use in the report).

Mac/Linux: same steps, e.g. `python src/extract_rag.py`, `python src/preprocess.py`, `python src/train.py`, `streamlit run app.py`.

## Rules the code enforces
- One dye per model; removal must lie in 0-100; duplicates dropped; missing values never filled.
- Models tuned with 5-fold CV on the training split only; scores reported on the held-out 20%.
- Test R2 above 0.98 triggers a leakage warning. Under 40 rows triggers a fallback warning.

## Deploy
Push to GitHub (commit `models/best_model.joblib`, `models/metrics.json`, `data/processed/`), then Streamlit Community Cloud -> `app.py`.
`requirements.txt` is the light set; the extraction packages are not needed there.

## Layout
```
data/papers/      PDFs in          data/raw/        one CSV per source/person
data/processed/   merged dataset   models/          best_model.joblib, metrics.json
src/              extract_rag.py, preprocess.py, train.py, make_demo_data.py
app.py            Streamlit app
```
Always open every DOI yourself before it goes in the report.
