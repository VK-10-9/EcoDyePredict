# 🌱 EcoDyePredict: ML-Based Prediction of Dye Removal from Wastewater

[![Python](https://img.shields.io/badge/Python-3.11%20%7C%203.12%20%7C%203.13-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.50+-FF4B4B.svg)](https://streamlit.io/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.4+-F7931E.svg)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-Enabled-008080.svg)](https://xgboost.readthedocs.io/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An end-to-end Machine Learning web application predicting how efficiently **Pumpkin Seed Husk (PSH)** (`Cucurbita pepo`) removes synthetic dyes from textile wastewater under various environmental and operational conditions.

---

## 🎯 What Does EcoDyePredict Do?

Textile and dyeing industries generate heavy wastewater containing synthetic dyes. Removing these pollutants usually requires expensive adsorbents and processes. **Pumpkin seed husk** is an agricultural byproduct that serves as a low-cost, sustainable biosorbent.

EcoDyePredict models the complex, non-linear relationships governing dye adsorption across 5 operational factors:

| Parameter | Unit | Description | Range Explored |
|---|---|---|---|
| **pH** | — | Acidity / basicity of wastewater solution | 2.0 – 12.0 |
| **PSH Dosage** | g/L | Amount of pumpkin seed husk added | 0.2 – 50.0 |
| **Initial Concentration** | mg/L | Starting concentration of the dye | 1.0 – 500.0 |
| **Contact Time** | min | Interaction duration in shaker/reactor | 5 – 2880 |
| **Temperature** | °C | Solution temperature during treatment | 20 – 50 |

**Output:** `removal_pct` — Predicted dye removal efficiency (%) with real-time target dosage optimization.

---

## 🏆 Model Performance & Benchmarks

All models are trained with rigorous 80/20 train/test splits, 5-fold cross-validation restricted strictly to training folds, and evaluated on held-out test data:

| Dye | Dye Class | Clean Rows | Best Algorithm | **Test $R^2$** | **Test RMSE** | **Test MAE** |
|---|---|:---:|:---:|:---:|:---:|:---:|
| **Erythrosine B** | Anionic (Acidic/Xanthene) | 101 | **XGBoost** | **0.884** | 9.21% | 5.56% |
| **Reactive Yellow 84** | Anionic (Reactive Azo) | 64 | **XGBoost** | **0.862** | 10.02% | 5.81% |
| **Basic Red 46** | Cationic (Azo) | 79 | **XGBoost** | **0.795** | 9.41% | 5.76% |
| **Reactive Black 5** | Anionic (Reactive Diazo) | 69 | **XGBoost** | **0.790** | 6.47% | 3.12% |
| **Basic Violet 10** | Cationic (Rhodamine B) | 81 | **XGBoost** | **0.712** | 10.40% | 7.78% |
| **Methylene Blue** | Cationic (Basic Phenothiazine) | 100 | **Linear Reg. / DT** | **0.790 / 0.663** | 5.54% | 4.75% |

---

## 💻 Interactive Streamlit Application Features

- **Tab 1: 🎯 Predict & Optimize** — Adjust experimental conditions with sliders bounded by research literature; get instant removal % and an operational dosage curve identifying the minimum dosage required to achieve your target removal (e.g. ≥ 85%).
- **Tab 2: 📊 Model Comparison** — Side-by-side benchmark comparison of Linear Regression, Decision Tree, Random Forest, and XGBoost (CV $R^2$, Test $R^2$, RMSE, MAE).
- **Tab 3: 🔍 Scientific Insights** — Permutation feature importance ranking and predicted vs. actual parity plot on held-out test data.
- **Tab 4: 📖 About & References** — Adsorption mechanisms, chemical classifications, and peer-reviewed DOI citations.
- **Sidebar** — Hardware and resource monitor + dye selector.

---

## 🚀 Quickstart Guide

### Option A: Windows (1-Click Run)
```cmd
# 1. First time setup (installs dependencies)
setup.bat

# 2. Train all models
2_build_dataset_and_train.bat

# 3. Launch Streamlit web app
3_run_app.bat
```

### Option B: macOS / Linux / Manual
```bash
# Clone the repository
git clone https://github.com/VK-10-9/EcoDyePredict.git
cd EcoDyePredict

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install requirements
pip install -r requirements.txt

# Train models for all dyes
python src/train_all_dyes.py

# Launch the app
streamlit run app.py
```

---

## 🤖 Optional: AI Paper Extraction (RAG)

If you add new research papers (`.pdf`) to `data/papers/`, you can extract experimental tables automatically using Google Gemini:

1. Copy `.env.example` to `.env`:
   ```bash
   cp .env.example .env
   ```
2. Add your free Gemini API key from [Google AI Studio](https://aistudio.google.com/apikey):
   ```env
   GEMINI_API_KEY=your_key_here
   ```
3. Run extraction:
   ```bash
   python src/extract_rag.py
   # Or on Windows: 1_extract_from_pdfs.bat
   ```

---

## 📓 Exploratory Data Analysis & Notebooks

An interactive Jupyter Notebook is provided in [`notebooks/EcoDyePredict_Analysis.ipynb`](notebooks/EcoDyePredict_Analysis.ipynb) featuring:
- Correlation matrices and distribution analysis
- Multimodel training and cross-validation
- Parity plots (actual vs. predicted)
- Dosage curve optimization

To run the notebook:
```bash
jupyter notebook notebooks/EcoDyePredict_Analysis.ipynb
```

---

## 📁 Repository Structure

```
EcoDyePredict/
├── app.py                      # Streamlit interactive web application
├── requirements.txt            # Python dependencies (lightweight, cloud-ready)
├── .env.example                # Template for Gemini API key
├── 0_demo_smoke_test.bat       # Synthetic data smoke test
├── 1_extract_from_pdfs.bat     # Automated RAG PDF extraction runner
├── 2_build_dataset_and_train.bat # Multi-dye dataset builder & model trainer
├── 3_run_app.bat               # Streamlit app launcher
├── src/
│   ├── compile_raw_data.py     # Compiles verified research data into raw CSVs
│   ├── extract_rag.py          # Gemini AI extraction over scientific papers
│   ├── preprocess.py           # Validates, deduplicates, and normalizes datasets
│   ├── train.py                # Hyperparameter tuning (GridSearchCV) & evaluation
│   └── train_all_dyes.py       # Batch trainer for all dyes & registry creator
├── data/
│   ├── raw/                    # Cleaned raw experimental data per dye
│   └── processed/              # Modeling splits and dataset summaries
├── models/
│   ├── dye_registry.json       # Registry linking dyes to their best model
│   ├── best_model.joblib       # Default active model (Methylene Blue)
│   └── <dye_slug>/             # Per-dye serialized models and metrics.json
└── notebooks/
    └── EcoDyePredict_Analysis.ipynb # Complete EDA and visualization notebook
```

---

## 🔬 Scientific Citations

1. **Apostol, L. C., Ghinea, C., Alves, M., & Gavrilescu, M. (2016)**. *Removal of Erythrosine B dye from water effluents using crop waste pumpkin seed hulls as adsorbent*. Desalination and Water Treatment, 57(48-49), 22585-22608. [DOI: 10.5004/dwt.2016.22585](https://doi.org/10.5004/dwt.2016.22585)
2. **Kowalkowska, A., & Jóźwiak, T. (2019)**. *Utilization of pumpkin (Cucurbita pepo) seed husks as a low-cost sorbent for removing anionic and cationic dyes from aqueous solutions*. Desalination and Water Treatment, 171, 397-407. [DOI: 10.5004/dwt.2019.24761](https://doi.org/10.5004/dwt.2019.24761)
3. **Küçük, M. (2024)**. *Investigation of Biosorption of Methylene Blue by Pumpkin Seed Shells*. Bitlis Eren University Journal of Science, 13(1), 84-94. [DOI: 10.17798/bitlisfen.1328584](https://doi.org/10.17798/bitlisfen.1328584)
4. **Aniagor, C. O., et al. (2024)**. *Removal of Methylene blue dye from contaminated wastewater using agricultural biomass*. Waste Management Bulletin.

---

## 📄 License

This project is licensed under the MIT License — see the LICENSE file for details.
