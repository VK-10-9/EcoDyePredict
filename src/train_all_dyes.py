"""
Preprocess and train models for every dye found in data/raw/*.csv
Saves individual model bundles in models/<dye_slug>/,
and sets the default dye in models/ (root).
"""
import json, pathlib, subprocess, sys
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
MODELS_DIR = ROOT / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# Find all dyes
csvs = [f for f in sorted(RAW_DIR.glob("*.csv")) if not f.name.upper().startswith("TEMPLATE")]
dfs = [pd.read_csv(f) for f in csvs]
all_df = pd.concat(dfs, ignore_index=True)
all_df["dye"] = all_df["dye"].astype(str).str.strip().str.lower()
dyes = all_df["dye"].value_counts().index.tolist()

print(f"Discovered {len(dyes)} dyes in data/raw: {dyes}")

registry = {}

for dye in dyes:
    slug = dye.replace(" ", "_")
    dye_proc = ROOT / "data" / "processed" / f"dataset_{slug}.csv"
    dye_models = MODELS_DIR / slug
    dye_models.mkdir(parents=True, exist_ok=True)
    
    print(f"\n==========================================")
    print(f"Processing & Training for Dye: {dye.upper()}")
    print(f"==========================================")
    
    # 1. Preprocess for this dye
    cmd_prep = [
        sys.executable, str(ROOT / "src" / "preprocess.py"),
        "--dye", dye,
        "--out", str(dye_proc)
    ]
    res_prep = subprocess.run(cmd_prep, capture_output=True, text=True)
    print(res_prep.stdout)
    if res_prep.returncode != 0:
        print(f"Preprocessing warning/skip for {dye}:", res_prep.stderr)
        continue
        
    # Check rows
    n_rows = len(pd.read_csv(dye_proc))
    print(f"Clean rows for {dye}: {n_rows}")
    
    # 2. Train for this dye
    cmd_train = [
        sys.executable, str(ROOT / "src" / "train.py"),
        "--data", str(dye_proc),
        "--out", str(dye_models)
    ]
    res_train = subprocess.run(cmd_train, capture_output=True, text=True)
    print(res_train.stdout)
    if res_train.returncode != 0:
        print(f"Training failed for {dye}:", res_train.stderr)
        continue
        
    metrics_path = dye_models / "metrics.json"
    if metrics_path.exists():
        met = json.loads(metrics_path.read_text())
        best_m = met.get("best", "Unknown")
        test_r2 = met["models"][best_m]["test_r2"]
        registry[dye] = {
            "slug": slug,
            "rows": n_rows,
            "best_model": best_m,
            "test_r2": test_r2,
            "models_dir": str(dye_models)
        }

# Save registry of all trained dyes
(MODELS_DIR / "dye_registry.json").write_text(json.dumps(registry, indent=2))
print("\n==========================================")
print("DYE REGISTRY COMPLETE:")
print(json.dumps(registry, indent=2))

# Also ensure default models/best_model.joblib and metrics.json are linked to the primary dye
if "methylene blue" in registry:
    primary = "methylene blue"
elif registry:
    primary = list(registry.keys())[0]
else:
    primary = None

if primary:
    import shutil
    prim_slug = registry[primary]["slug"]
    shutil.copy(MODELS_DIR / prim_slug / "best_model.joblib", MODELS_DIR / "best_model.joblib")
    shutil.copy(MODELS_DIR / prim_slug / "metrics.json", MODELS_DIR / "metrics.json")
    print(f"\nDefault active model set to: {primary} ({prim_slug})")
