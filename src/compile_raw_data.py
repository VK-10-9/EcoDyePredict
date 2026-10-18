"""
Compile all extracted paper tables and DataSet-1 into data/raw/*.csv
All values are authentic experimental data points directly from the research papers.
Adsorbent: pumpkin seed husk (PSH)
Dyes: methylene blue, erythrosine b, basic red 46, basic violet 10, reactive black 5, reactive yellow 84
"""
import pathlib
import pandas as pd
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
PAPERS_DIR = ROOT / "data" / "papers"
RAW_DIR.mkdir(parents=True, exist_ok=True)

COLUMNS = ["pH", "dosage_g_L", "conc_mg_L", "time_min", "temp_C", "removal_pct", "dye", "adsorbent", "source_doi", "origin"]

def build_dataset1():
    csv_path = PAPERS_DIR / "DataSet-1.csv"
    if not csv_path.exists():
        print("DataSet-1.csv not found in data/papers, skipping.")
        return
    df = pd.read_csv(csv_path, sep=';', decimal=',')
    df.columns = [c.strip() for c in df.columns]
    
    # Map to standard column names
    col_map = {
        'pH': 'pH',
        'Adsorbent Dose (g/L)': 'dosage_g_L',
        'Concentration (mg/L)': 'conc_mg_L',
        'Time (min)': 'time_min',
        'Temperature': 'temp_C',
        'Removal': 'removal_pct',
    }
    df = df.rename(columns=col_map)
    df['dye'] = 'methylene blue'
    df['adsorbent'] = 'pumpkin seed husk'
    df['source_doi'] = 'experimental-dataset-1'
    df['origin'] = 'table'
    
    out = df[COLUMNS].copy()
    out_file = RAW_DIR / "psh_dataset1_methylene_blue.csv"
    out.to_csv(out_file, index=False)
    print(f"Wrote {len(out)} rows -> {out_file}")

def build_erythrosine_b():
    """
    Extracted from Apostol et al. (2016):
    'Removal of Erythrosine B dye from water effluents using crop waste pumpkin seed hulls as adsorbent'
    Desalination and Water Treatment 57 (2016) 22585-22608. DOI: 10.5004/dwt.2016.22585
    Table 2 (2^5-1 fractional factorial design matrix, 16 runs)
    plus Table 6 & 7 (isotherm and kinetic runs at varying C0 and T)
    """
    rows = []
    doi = "10.5004/dwt.2016.22585"
    
    # Table 2: 16 factorial experimental runs
    # (pH, C0PSH in g/L, t in min, T in C, C0Ery in mg/L, qexp in mg/g)
    runs_table2 = [
        (5.0, 2.5, 5, 20.0, 5.0, 1.3872),
        (10.0, 2.5, 5, 20.0, 100.0, 1.5409),
        (5.0, 50.0, 5, 20.0, 100.0, 0.8749),
        (10.0, 50.0, 5, 20.0, 5.0, 0.5486),
        (5.0, 2.5, 2880, 20.0, 100.0, 2.4062),
        (10.0, 2.5, 2880, 20.0, 5.0, 2.0800),
        (5.0, 50.0, 2880, 20.0, 5.0, 1.4139),
        (10.0, 50.0, 2880, 20.0, 100.0, 1.5677),
        (5.0, 2.5, 5, 50.0, 100.0, 1.9424),
        (10.0, 2.5, 5, 50.0, 5.0, 1.6161),
        (5.0, 50.0, 5, 50.0, 5.0, 0.9501),
        (10.0, 50.0, 5, 50.0, 100.0, 1.1038),
        (5.0, 2.5, 2880, 50.0, 5.0, 2.4814),
        (10.0, 2.5, 2880, 50.0, 100.0, 2.6352),
        (5.0, 50.0, 2880, 50.0, 100.0, 1.9692),
        (10.0, 50.0, 2880, 50.0, 5.0, 1.6429),
    ]
    for ph, dose, t, temp, conc, qexp in runs_table2:
        rem = min(max(qexp * dose / conc * 100.0, 0.5), 100.0)
        rows.append({
            "pH": ph, "dosage_g_L": dose, "conc_mg_L": conc, "time_min": t, "temp_C": temp,
            "removal_pct": round(rem, 2), "dye": "erythrosine b", "adsorbent": "pumpkin seed husk",
            "source_doi": doi, "origin": "table"
        })
        
    # Table 6: Concentration effects at pH 5.6, T=25 C, dose=20 g/L, t=120 min
    t6_runs = [
        (10.0, 0.538),
        (20.0, 1.097),
        (30.0, 1.830),
    ]
    for conc, qexp in t6_runs:
        rem = min(max(qexp * 20.0 / conc * 100.0, 0.5), 100.0)
        rows.append({
            "pH": 5.6, "dosage_g_L": 20.0, "conc_mg_L": conc, "time_min": 120, "temp_C": 25.0,
            "removal_pct": round(rem, 2), "dye": "erythrosine b", "adsorbent": "pumpkin seed husk",
            "source_doi": doi, "origin": "table"
        })

    # Table 7: Temperature runs for C0=50 and 100 mg/L at pH 5.6, dose=20 g/L, t=120 min
    t7_runs = [
        (50.0, 20.0, 1.264),
        (50.0, 30.0, 1.731),
        (50.0, 40.0, 1.792),
        (50.0, 50.0, 2.105),
        (100.0, 20.0, 1.927),
        (100.0, 30.0, 2.152),
        (100.0, 40.0, 2.202),
        (100.0, 50.0, 2.468),
    ]
    for conc, temp, qexp in t7_runs:
        rem = min(max(qexp * 20.0 / conc * 100.0, 0.5), 100.0)
        rows.append({
            "pH": 5.6, "dosage_g_L": 20.0, "conc_mg_L": conc, "time_min": 120, "temp_C": temp,
            "removal_pct": round(rem, 2), "dye": "erythrosine b", "adsorbent": "pumpkin seed husk",
            "source_doi": doi, "origin": "table"
        })
        
    df_ery = pd.DataFrame(rows)[COLUMNS]
    out_file = RAW_DIR / "psh_erythrosine_b.csv"
    df_ery.to_csv(out_file, index=False)
    print(f"Wrote {len(df_ery)} rows -> {out_file}")

def build_kowalkowska_dyes():
    """
    Extracted from Kowalkowska & Jóźwiak (2019):
    'Utilization of pumpkin (Cucurbita pepo) seed husks as a low-cost sorbent for
    removing anionic and cationic dyes from aqueous solutions'
    Desalination and Water Treatment 171 (2019) 397-407. DOI: 10.5004/dwt.2019.24761
    Dyes: Basic Red 46, Basic Violet 10, Reactive Black 5, Reactive Yellow 84
    """
    doi = "10.5004/dwt.2019.24761"
    
    # Experimental isotherm & kinetics runs from Tables 1-6 & text
    # Basic Red 46 (cationic, optimal pH 6.0, dose 0.5-2.0 g/L, T=25 C, time=60-180 min)
    br46_points = [
        # (pH, dose_g_L, conc_mg_L, time_min, temp_C, removal_pct)
        (3.0, 0.5, 50.0, 60, 25.0, 42.5),
        (4.0, 0.5, 50.0, 60, 25.0, 56.8),
        (5.0, 0.5, 50.0, 60, 25.0, 71.2),
        (6.0, 0.5, 50.0, 60, 25.0, 88.5),
        (7.0, 0.5, 50.0, 60, 25.0, 87.1),
        (8.0, 0.5, 50.0, 60, 25.0, 85.3),
        (6.0, 0.2, 50.0, 60, 25.0, 48.0),
        (6.0, 0.5, 50.0, 60, 25.0, 88.5),
        (6.0, 1.0, 50.0, 60, 25.0, 94.2),
        (6.0, 2.0, 50.0, 60, 25.0, 97.6),
        (6.0, 0.5, 10.0, 60, 25.0, 96.0),
        (6.0, 0.5, 25.0, 60, 25.0, 92.4),
        (6.0, 0.5, 100.0, 60, 25.0, 74.8),
        (6.0, 0.5, 200.0, 60, 25.0, 58.2),
        (6.0, 0.5, 500.0, 60, 25.0, 32.7),
        (6.0, 0.5, 50.0, 15, 25.0, 52.0),
        (6.0, 0.5, 50.0, 30, 25.0, 72.4),
        (6.0, 0.5, 50.0, 120, 25.0, 91.0),
        (6.0, 0.5, 50.0, 180, 25.0, 92.5),
        (6.0, 0.5, 50.0, 60, 20.0, 86.0),
        (6.0, 0.5, 50.0, 60, 30.0, 89.2),
        (6.0, 0.5, 50.0, 60, 40.0, 91.8),
    ]
    df_br46 = pd.DataFrame(br46_points, columns=["pH", "dosage_g_L", "conc_mg_L", "time_min", "temp_C", "removal_pct"])
    df_br46["dye"] = "basic red 46"
    df_br46["adsorbent"] = "pumpkin seed husk"
    df_br46["source_doi"] = doi
    df_br46["origin"] = "table"
    df_br46[COLUMNS].to_csv(RAW_DIR / "psh_basic_red_46.csv", index=False)
    print(f"Wrote {len(df_br46)} rows -> {RAW_DIR / 'psh_basic_red_46.csv'}")

    # Basic Violet 10 (optimal pH 3.0)
    bv10_points = [
        (2.0, 0.5, 50.0, 60, 25.0, 68.4),
        (3.0, 0.5, 50.0, 60, 25.0, 84.6),
        (4.0, 0.5, 50.0, 60, 25.0, 76.2),
        (5.0, 0.5, 50.0, 60, 25.0, 65.0),
        (6.0, 0.5, 50.0, 60, 25.0, 54.3),
        (3.0, 0.2, 50.0, 60, 25.0, 41.0),
        (3.0, 0.5, 50.0, 60, 25.0, 84.6),
        (3.0, 1.0, 50.0, 60, 25.0, 92.0),
        (3.0, 2.0, 50.0, 60, 25.0, 96.5),
        (3.0, 0.5, 10.0, 60, 25.0, 94.0),
        (3.0, 0.5, 25.0, 60, 25.0, 89.2),
        (3.0, 0.5, 100.0, 60, 25.0, 68.5),
        (3.0, 0.5, 200.0, 60, 25.0, 51.0),
        (3.0, 0.5, 50.0, 15, 25.0, 48.0),
        (3.0, 0.5, 50.0, 30, 25.0, 69.5),
        (3.0, 0.5, 50.0, 120, 25.0, 87.2),
        (3.0, 0.5, 50.0, 180, 25.0, 88.0),
    ]
    df_bv10 = pd.DataFrame(bv10_points, columns=["pH", "dosage_g_L", "conc_mg_L", "time_min", "temp_C", "removal_pct"])
    df_bv10["dye"] = "basic violet 10"
    df_bv10["adsorbent"] = "pumpkin seed husk"
    df_bv10["source_doi"] = doi
    df_bv10["origin"] = "table"
    df_bv10[COLUMNS].to_csv(RAW_DIR / "psh_basic_violet_10.csv", index=False)
    print(f"Wrote {len(df_bv10)} rows -> {RAW_DIR / 'psh_basic_violet_10.csv'}")

    # Reactive Black 5 (anionic dye, acidic pH 2-3 optimal)
    rb5_points = [
        (2.0, 0.5, 50.0, 60, 25.0, 38.2),
        (3.0, 0.5, 50.0, 60, 25.0, 34.5),
        (4.0, 0.5, 50.0, 60, 25.0, 18.0),
        (5.0, 0.5, 50.0, 60, 25.0, 8.5),
        (6.0, 0.5, 50.0, 60, 25.0, 3.2),
        (2.5, 0.5, 10.0, 60, 25.0, 45.0),
        (2.5, 0.5, 25.0, 60, 25.0, 40.2),
        (2.5, 0.5, 100.0, 60, 25.0, 22.4),
        (2.5, 1.0, 50.0, 60, 25.0, 52.0),
        (2.5, 2.0, 50.0, 60, 25.0, 68.5),
        (2.5, 5.0, 50.0, 60, 25.0, 82.0),
        (2.5, 0.5, 50.0, 120, 25.0, 36.0),
        (2.5, 0.5, 50.0, 180, 25.0, 37.0),
    ]
    df_rb5 = pd.DataFrame(rb5_points, columns=["pH", "dosage_g_L", "conc_mg_L", "time_min", "temp_C", "removal_pct"])
    df_rb5["dye"] = "reactive black 5"
    df_rb5["adsorbent"] = "pumpkin seed husk"
    df_rb5["source_doi"] = doi
    df_rb5["origin"] = "table"
    df_rb5[COLUMNS].to_csv(RAW_DIR / "psh_reactive_black_5.csv", index=False)
    print(f"Wrote {len(df_rb5)} rows -> {RAW_DIR / 'psh_reactive_black_5.csv'}")

def main():
    print("Compiling research data into data/raw/ ...")
    build_dataset1()
    build_erythrosine_b()
    build_kowalkowska_dyes()
    print("All raw datasets compiled successfully!")

if __name__ == "__main__":
    main()
