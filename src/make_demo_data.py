"""
SYNTHETIC demo data - ONLY to smoke-test the pipeline and app before you have real data.
It is NOT real adsorption data. Never use it in the report. Output: data/demo/demo_raw.csv
"""
import pathlib
import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parent.parent
rng = np.random.default_rng(0)
n = 80
pH = rng.uniform(2, 10, n)
dose = rng.uniform(0.5, 6, n)
conc = rng.uniform(10, 200, n)
time = rng.uniform(5, 180, n)
temp = rng.uniform(20, 50, n)
r = (30 + 8 * np.log(dose + 1) * 3 - 0.15 * conc + 12 * (1 - np.exp(-time / 40))
     + 2.5 * (pH - 2) + 0.2 * temp + rng.normal(0, 3, n))
df = pd.DataFrame({"pH": pH.round(1), "dosage_g_L": dose.round(2), "conc_mg_L": conc.round(0),
                   "time_min": time.round(0), "temp_C": temp.round(0),
                   "removal_pct": np.clip(r, 1, 99).round(1),
                   "dye": "demo dye", "adsorbent": "demo adsorbent", "source_doi": "SYNTHETIC-DEMO"})
(ROOT / "data" / "demo").mkdir(parents=True, exist_ok=True)
df.to_csv(ROOT / "data" / "demo" / "demo_raw.csv", index=False)
print("wrote data/demo/demo_raw.csv (SYNTHETIC)")
