import json, os, pathlib
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import psutil

ROOT = pathlib.Path(__file__).resolve().parent
MODELS_ROOT = ROOT / os.environ.get("ECODYE_MODELS", "models")
REGISTRY_FILE = MODELS_ROOT / "dye_registry.json"

LABELS = {
    "pH": "pH",
    "dosage_g_L": "PSH dosage (g/L)",
    "conc_mg_L": "Initial dye concentration (mg/L)",
    "time_min": "Contact time (min)",
    "temp_C": "Temperature (°C)"
}

DYE_INFO = {
    "methylene blue": {
        "class": "Cationic (Basic)",
        "optimal_pH": "High pH (7 - 10)",
        "mechanism": "Deprotonation of PSH functional groups (hydroxyl & carboxyl) creates negative surface charge, strongly attracting cationic MB+ ions.",
        "papers": ["Aniagor et al. (2024), Waste Management Bulletin", "Küçük (2024), Bitlis Eren Univ. J. Sci."]
    },
    "erythrosine b": {
        "class": "Anionic (Acidic/Xanthene)",
        "optimal_pH": "Low to Neutral pH (4 - 5.6)",
        "mechanism": "At acidic pH, surface functional groups on PSH protonate, attracting negatively charged Erythrosine B sulfonate/phenolate groups.",
        "papers": ["Apostol, Ghinea, Alves & Gavrilescu (2016), Desalination & Water Treatment"]
    },
    "basic red 46": {
        "class": "Cationic (Azo)",
        "optimal_pH": "pH 6.0",
        "mechanism": "Electrostatic attraction between cationic azo group and ionized surface active sites of pumpkin seed husks.",
        "papers": ["Kowalkowska & Jóźwiak (2019), Desalination & Water Treatment"]
    },
    "basic violet 10": {
        "class": "Cationic (Rhodamine B)",
        "optimal_pH": "pH 3.0",
        "mechanism": "Sorption favored at moderate acidity through hydrogen bonding and pi-pi stacking with aromatic lignin structures.",
        "papers": ["Kowalkowska & Jóźwiak (2019), Desalination & Water Treatment"]
    },
    "reactive black 5": {
        "class": "Anionic (Reactive Diazo)",
        "optimal_pH": "Acidic pH (2 - 3)",
        "mechanism": "Strong protonation of adsorbent amine/hydroxyl groups at pH 2-3 allows electrostatic binding of anionic sulfonate groups.",
        "papers": ["Kowalkowska & Jóźwiak (2019), Desalination & Water Treatment"]
    },
    "reactive yellow 84": {
        "class": "Anionic (Reactive Azo)",
        "optimal_pH": "Acidic pH (2 - 3)",
        "mechanism": "Electrostatic attraction between protonated PSH surface hydroxyl sites and anionic sulfonic groups of RY84 dye.",
        "papers": ["Kowalkowska & Jóźwiak (2019), Desalination & Water Treatment"]
    }
}

st.set_page_config(page_title="EcoDyePredict · Pumpkin Seed Husk", page_icon="🌱", layout="wide")

# Sidebar: System Health & Dye Selection
with st.sidebar:
    st.header("🖥️ System Health Monitor")
    
    # Live PC Health
    mem = psutil.virtual_memory()
    cpu_pct = psutil.cpu_percent(interval=0.1)
    avail_gb = mem.available / (1024**3)
    total_gb = mem.total / (1024**3)
    
    st.caption(f"Hardware: **{psutil.cpu_count(logical=True)} CPU Threads** · **{total_gb:.1f} GB RAM**")
    st.progress(mem.percent / 100.0, text=f"RAM Used: {mem.percent}% ({avail_gb:.2f} GB free of {total_gb:.1f} GB)")
    st.progress(min(cpu_pct / 100.0, 1.0), text=f"CPU Load: {cpu_pct:.1f}%")
    
    st.divider()
    st.header("🧪 Dye & Model Selection")
    
    # Load Registry
    if REGISTRY_FILE.exists():
        registry = json.loads(REGISTRY_FILE.read_text())
        dye_options = list(registry.keys())
    else:
        registry = {}
        dye_options = ["methylene blue"]

    selected_dye = st.selectbox(
        "Choose Dye to Model:",
        options=dye_options,
        index=0,
        format_func=lambda d: d.title()
    )
    
    dye_slug = selected_dye.replace(" ", "_")
    dye_model_dir = MODELS_ROOT / dye_slug
    if not dye_model_dir.exists():
        dye_model_dir = MODELS_ROOT

    dye_meta = DYE_INFO.get(selected_dye, {
        "class": "Biomass Adsorption",
        "optimal_pH": "Standard",
        "mechanism": "Surface adsorption onto pumpkin seed husk biomacromolecules.",
        "papers": ["PSH Literature"]
    })
    
    st.info(f"**Dye Class:** {dye_meta['class']}\n\n**Optimal pH:** {dye_meta['optimal_pH']}")


@st.cache_resource
def load_dye_model(folder_path):
    folder = pathlib.Path(folder_path)
    model_file = folder / "best_model.joblib"
    metrics_file = folder / "metrics.json"
    if not model_file.exists():
        model_file = MODELS_ROOT / "best_model.joblib"
        metrics_file = MODELS_ROOT / "metrics.json"
    return joblib.load(model_file), json.loads(metrics_file.read_text())


if not (MODELS_ROOT / "best_model.joblib").exists():
    st.title("🌱 EcoDyePredict")
    st.error("No trained models found. Run `python src/train_all_dyes.py` first.")
    st.stop()

model, M = load_dye_model(str(dye_model_dir))
F = M["features"]

st.title("🌱 EcoDyePredict")
st.caption(f"Predicting dye removal (%) by **Pumpkin Seed Husk (PSH)** · Active Model Dye: **{selected_dye.upper()}** (Best: {M['best']})")

if M.get("demo"):
    st.warning("DEMO MODE: trained on SYNTHETIC data.")


def predict(vals):
    p = float(model.predict(pd.DataFrame([vals], columns=F))[0])
    return min(max(p, 0.0), 100.0)


tab1, tab2, tab3, tab4 = st.tabs(["🎯 Predict & Optimize", "📊 Model Comparison", "🔍 Scientific Insights", "📖 About & References"])

with tab1:
    st.subheader(f"Screen Adsorption Conditions for {selected_dye.title()}")
    st.markdown(
        f"**Adsorption Mechanism:** {dye_meta['mechanism']}\n\n"
        "*Slider bounds are clamped strictly to the experimental domain observed in the research dataset to guarantee valid physical predictions.*"
    )
    
    cols = st.columns(5)
    vals = {}
    for c, f in zip(cols, F):
        lo, hi = M["ranges"][f]
        with c:
            if lo == hi:
                st.metric(LABELS[f], f"{lo:g}", help="Fixed condition in this study")
                vals[f] = lo
            else:
                step_val = float(max((hi - lo) / 100.0, 0.01))
                mid_val = float((lo + hi) / 2.0)
                vals[f] = st.slider(LABELS[f], float(lo), float(hi), mid_val, step=step_val)
                
    pred = predict(vals)
    
    st.write("")
    m_col1, m_col2 = st.columns([1, 2])
    with m_col1:
        st.metric("Predicted Removal Efficiency", f"{pred:.2f} %")
    with m_col2:
        st.write("Removal Gauge")
        st.progress(int(round(pred)))

    st.divider()
    st.subheader("Lowest Dosage Reaching Target Removal")
    col_t1, col_t2 = st.columns([1, 2])
    with col_t1:
        target = st.slider("Target Removal (%)", 10, 99, 85)
    
    lo, hi = M["ranges"]["dosage_g_L"]
    grid = np.linspace(lo, hi, 150)
    preds = [predict({**vals, "dosage_g_L": float(d)}) for d in grid]
    ok = [d for d, p in zip(grid, preds) if p >= target]
    
    with col_t2:
        if ok:
            st.success(f"Optimal operational dosage: **{ok[0]:.2f} g/L** reaches ≥ **{target}%** removal at these conditions.")
        else:
            st.warning(f"No dosage in the tested range ({lo:g} – {hi:g} g/L) achieves {target}% removal under current conditions.")
            
    fig_dose = px.line(
        x=grid, y=preds,
        labels={"x": LABELS["dosage_g_L"], "y": "Predicted Removal (%)"},
        title=f"Removal vs. PSH Dosage for {selected_dye.title()}"
    )
    fig_dose.add_hline(y=target, line_dash="dash", line_color="red", annotation_text=f"Target {target}%")
    st.plotly_chart(fig_dose, use_container_width=True)

with tab2:
    st.subheader(f"Model Benchmarking for {selected_dye.title()}")
    rows = [
        {
            "Model": k,
            "CV R² (Train 5-Fold)": round(v["cv_r2"], 3),
            "Test R² (Held-out)": round(v["test_r2"], 3),
            "Test RMSE (%)": round(v["test_rmse"], 2),
            "Test MAE (%)": round(v["test_mae"], 2),
            "Status": "🏆 Active Best" if k == M["best"] else ""
        }
        for k, v in M["models"].items()
    ]
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
    st.caption(f"Trained on {M['n_train']} split rows, evaluated on {M['n_test']} held-out test rows. Best model chosen by cross-validated R².")
    
    for w in M.get("warnings", []):
        st.warning(f"⚠️ {w}")

with tab3:
    st.subheader(f"Feature Importance & Residual Analysis for {selected_dye.title()}")
    col_a, col_b = st.columns(2)
    
    imp_df = pd.DataFrame({
        "Feature": [LABELS[f] for f in F],
        "Permutation Importance (Drop in R²)": [M["importance"].get(f, 0.0) for f in F]
    }).sort_values("Permutation Importance (Drop in R²)")
    
    fig_imp = px.bar(
        imp_df,
        x="Permutation Importance (Drop in R²)",
        y="Feature",
        orientation="h",
        title="Feature Importance Ranking"
    )
    col_a.plotly_chart(fig_imp, use_container_width=True)
    
    act = M["test_predictions"]["actual"]
    prd = M["test_predictions"]["predicted"]
    fig_res = go.Figure()
    fig_res.add_trace(go.Scatter(x=act, y=prd, mode="markers", name="Test Data Points", marker=dict(size=9, color="#10b981")))
    fig_res.add_trace(go.Scatter(x=[0, 100], y=[0, 100], mode="lines", name="1:1 Perfect Fit", line=dict(dash="dash", color="#ef4444")))
    fig_res.update_layout(
        title="Predicted vs. Actual Removal on Held-Out Test Set",
        xaxis_title="Actual Removal (%)",
        yaxis_title="Predicted Removal (%)"
    )
    col_b.plotly_chart(fig_res, use_container_width=True)

with tab4:
    st.subheader("About EcoDyePredict & Provenance")
    st.markdown(f"""
**Core Goal:** EcoDyePredict allows environmental researchers and wastewater engineers to simulate the adsorption efficiency of **pumpkin seed husk (PSH)** (`Cucurbita pepo`) without requiring costly, time-consuming wet lab trials.

### Active Dye Summary: **{selected_dye.title()}**
* **Dye Class:** {dye_meta['class']}
* **Optimal pH:** {dye_meta['optimal_pH']}
* **Physical Mechanism:** {dye_meta['mechanism']}
* **Key Paper Citations:**
""")
    for p in dye_meta['papers']:
        st.markdown(f"- *{p}*")

    st.markdown(f"""
### Dataset Statistics:
* **Total Clean Rows for this Dye:** {M['n_rows']}
* **Trained Models:** Linear Regression, Decision Tree Regressor, Random Forest Regressor
* **Validation Strategy:** 80/20 train/test split with cross-validation restricted strictly to training folds.
""")
    
    dataset_file = ROOT / "data" / "processed" / f"dataset_{dye_slug}.csv"
    if dataset_file.exists():
        st.subheader("Provenance & Sources")
        d = pd.read_csv(dataset_file)
        st.dataframe(d["source_doi"].value_counts().rename_axis("Source Identifier / DOI").reset_index(name="Rows"), hide_index=True)
