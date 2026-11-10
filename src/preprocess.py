"""
Step 2a - merge every CSV in data/raw/, validate, lock ONE dye, write data/processed/dataset.csv

Usage:
  python src/preprocess.py                      # auto-picks the dye with most complete rows
  python src/preprocess.py --dye "methylene blue"
  python src/preprocess.py --include-graph-estimates
"""
import argparse, json, pathlib, sys
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parent.parent
FEATS = ["pH", "dosage_g_L", "conc_mg_L", "time_min", "temp_C"]
TARGET = "removal_pct"
ALL = FEATS + [TARGET, "dye", "adsorbent", "source_doi", "origin"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default=str(ROOT / "data" / "raw"))
    ap.add_argument("--out", default=str(ROOT / "data" / "processed" / "dataset.csv"))
    ap.add_argument("--dye", default=None, help="dye to model (exact name as in the CSVs, case-insensitive)")
    ap.add_argument("--include-graph-estimates", action="store_true")
    a = ap.parse_args()

    files = [f for f in sorted(pathlib.Path(a.raw).glob("*.csv")) if not f.name.upper().startswith("TEMPLATE")]
    if not files:
        sys.exit(f"No CSV files in {a.raw}")
    parts = []
    for f in files:
        d = pd.read_csv(f)
        d.columns = [c.strip() for c in d.columns]
        d["file"] = f.name
        parts.append(d)
        print(f"read {f.name}: {len(d)} rows")
    df = pd.concat(parts, ignore_index=True)
    for c in ALL:
        if c not in df.columns:
            df[c] = pd.NA
    df["origin"] = df["origin"].fillna("manual")
    for c in FEATS + [TARGET]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    for c in ["dye", "adsorbent"]:
        df[c] = df[c].astype("string").str.strip().str.lower()
    df["dye"] = df["dye"].replace({
        "methylene blue (mb)": "methylene blue",
        "mb": "methylene blue",
        "rhodamine b": "basic violet 10",
        "rhodamine b (bv10)": "basic violet 10",
    })
    n0 = len(df)
    log = {"rows_read": n0}

    if not a.include_graph_estimates:
        m = df["origin"].eq("graph_estimate")
        log["dropped_graph_estimate"] = int(m.sum())
        df = df[~m]

    m = df[FEATS + [TARGET, "dye"]].isna().any(axis=1)
    log["dropped_missing_values"] = int(m.sum())
    df = df[~m]

    bad = ~df[TARGET].between(0, 100) | ~df["pH"].between(0, 14) | (df[["dosage_g_L", "conc_mg_L", "time_min"]] <= 0).any(axis=1)
    log["dropped_out_of_range"] = int(bad.sum())
    df = df[~bad]

    dup = df.duplicated(subset=FEATS + [TARGET, "dye"])
    log["dropped_duplicates"] = int(dup.sum())
    df = df[~dup]

    if df.empty:
        sys.exit("No valid rows left after cleaning. Check your CSVs.\n" + json.dumps(log, indent=2))

    counts = df["dye"].value_counts()
    print("\nClean rows per dye:\n" + counts.to_string())
    dye = (a.dye or counts.index[0]).strip().lower()
    if dye not in counts.index:
        sys.exit(f"Dye '{dye}' not found. Available: {list(counts.index)}")
    df = df[df["dye"] == dye].copy()
    log["dye"] = dye
    log["rows_final"] = len(df)
    log["adsorbents"] = sorted(df["adsorbent"].dropna().unique().tolist())
    log["n_sources"] = int(df["source_doi"].nunique())

    const = [c for c in FEATS if df[c].nunique() <= 1]
    log["constant_features"] = const

    out = pathlib.Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df[ALL].to_csv(out, index=False)
    (out.parent / (out.stem + "_summary.json")).write_text(json.dumps(log, indent=2))

    print(f"\nLocked dye: {dye} | final rows: {len(df)} | sources: {log['n_sources']}")
    print(json.dumps({k: v for k, v in log.items() if k.startswith("dropped")}, indent=2))
    if const:
        print(f"WARNING: constant features (no variation, cannot be learned): {const}")
    if len(df) < 40:
        print("WARNING: under 40 rows. Per your brief: use the Zenodo fallback for the pipeline "
              "and present PSH as a case study, or extract more rows.")
    if df["adsorbent"].nunique() > 1:
        print(f"WARNING: several adsorbent forms in the dataset: {log['adsorbents']}")
    print(f"Saved -> {out}")


if __name__ == "__main__":
    main()
