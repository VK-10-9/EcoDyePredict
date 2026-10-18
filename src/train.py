"""
Step 2b - train Linear Regression, Decision Tree, Random Forest (+ XGBoost if installed).
80/20 split, 5-fold CV tuning on the TRAIN part only, report R2 / RMSE / MAE on the held-out test set.
Best model (by cross-validated R2) -> models/best_model.joblib, plus models/metrics.json
"""
import argparse, json, pathlib, sys, warnings
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

warnings.filterwarnings("ignore")
ROOT = pathlib.Path(__file__).resolve().parent.parent
FEATS = ["pH", "dosage_g_L", "conc_mg_L", "time_min", "temp_C"]
TARGET = "removal_pct"
SEED = 42


def candidates():
    c = {
        "Linear Regression": (Pipeline([("sc", StandardScaler()), ("m", LinearRegression())]),
                              {"m__fit_intercept": [True]}),
        "Decision Tree": (DecisionTreeRegressor(random_state=SEED),
                          {"max_depth": [2, 3, 4, 5], "min_samples_leaf": [1, 2, 4]}),
        "Random Forest": (RandomForestRegressor(random_state=SEED),
                          {"n_estimators": [100, 200, 400], "max_depth": [3, 5, 8],
                           "min_samples_leaf": [1, 2, 3]}),
    }
    try:
        from xgboost import XGBRegressor
        c["XGBoost"] = (XGBRegressor(random_state=SEED, verbosity=0),
                        {"n_estimators": [100, 200], "max_depth": [2, 3, 4], "learning_rate": [0.05, 0.1]})
    except Exception:
        print("xgboost not installed - skipping (optional model).")
    return c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT / "data" / "processed" / "dataset.csv"))
    ap.add_argument("--out", default=str(ROOT / "models"))
    ap.add_argument("--demo", action="store_true", help="mark outputs as synthetic demo data")
    a = ap.parse_args()

    df = pd.read_csv(a.data)
    n = len(df)
    if n < 10:
        sys.exit(f"Only {n} rows - too few to train honestly. Get more data first.")
    X, y = df[FEATS], df[TARGET]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=SEED)
    cv = KFold(n_splits=min(5, max(2, len(Xtr))), shuffle=True, random_state=SEED)

    results, fitted = {}, {}
    for name, (est, grid) in candidates().items():
        gs = GridSearchCV(est, grid, cv=cv, scoring="r2", n_jobs=-1).fit(Xtr, ytr)
        p = gs.best_estimator_.predict(Xte)
        results[name] = {
            "cv_r2": float(gs.best_score_),
            "test_r2": float(r2_score(yte, p)),
            "test_rmse": float(np.sqrt(mean_squared_error(yte, p))),
            "test_mae": float(mean_absolute_error(yte, p)),
            "best_params": {k: (v if isinstance(v, (int, float, str, bool, type(None))) else str(v))
                            for k, v in gs.best_params_.items()},
        }
        fitted[name] = gs.best_estimator_
        r = results[name]
        print(f"{name:18s} cvR2={r['cv_r2']:.3f}  testR2={r['test_r2']:.3f}  RMSE={r['test_rmse']:.2f}  MAE={r['test_mae']:.2f}")

    best = max(results, key=lambda k: results[k]["cv_r2"])
    model = fitted[best]
    pred = model.predict(Xte)
    Xi, yi = (Xte, yte) if len(Xte) >= 5 else (X, y)
    imp = permutation_importance(model, Xi, yi, n_repeats=30, random_state=SEED, scoring="r2")
    importance = {f: float(v) for f, v in zip(FEATS, imp.importances_mean)}

    warn = []
    if results[best]["test_r2"] > 0.98:
        warn.append("Test R2 > 0.98: suspect leakage or duplicate rows, not a great model.")
    if n < 40:
        warn.append(f"Only {n} rows: scores are unstable.")
    for w in warn:
        print("WARNING:", w)

    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, out / "best_model.joblib")
    metrics = {
        "demo": bool(a.demo),
        "dye": str(df["dye"].iloc[0]) if "dye" in df else None,
        "n_rows": n, "n_train": len(Xtr), "n_test": len(Xte),
        "features": FEATS,
        "ranges": {f: [float(X[f].min()), float(X[f].max())] for f in FEATS},
        "models": results, "best": best,
        "importance": importance,
        "test_predictions": {"actual": yte.tolist(), "predicted": [float(v) for v in pred]},
        "warnings": warn,
    }
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"\nBest model: {best}  ->  {out / 'best_model.joblib'}")


if __name__ == "__main__":
    main()
