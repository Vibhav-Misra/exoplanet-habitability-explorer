import json
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (classification_report, confusion_matrix,
                             f1_score, precision_recall_curve, roc_auc_score,
                             roc_curve, average_precision_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier

DATA_PARQUET = Path("data/latest.parquet")
MODELS_DIR = Path("models")
MODELS_DIR.mkdir(exist_ok=True)

# --- Label definition (tweak if you like) ---
# Optimistic Habitable Zone proxy:
#   - Insolation: 0.35 to 1.5 Earth=1
#   - Radius: 0.5 to 1.75 Earth radii (rocky-ish)
HZ_BOUNDS = {
    "insolation": (0.35, 1.5),
    "radius_Re": (0.5, 1.75),
}

# Columns (use raw names from fetch_data to avoid confusion)
NUM_FEATURES = [
    "pl_eqt",     # equilibrium temp
    "st_teff",    # stellar temp
    "st_rad",     # stellar radius
    "st_mass",    # stellar mass
    "sy_dist",    # distance
    "pl_orbper",  # period
]
CAT_FEATURES = ["discoverymethod"]

CAT_FEATURES = [
    "discoverymethod"
]

def make_label(df: pd.DataFrame) -> pd.Series:
    rmin, rmax = HZ_BOUNDS["radius_Re"]
    imin, imax = HZ_BOUNDS["insolation"]
    radius_ok = df["pl_rade"].between(rmin, rmax, inclusive="both")
    ins_ok = df["pl_insol"].between(imin, imax, inclusive="both")
    # label = 1 only if both are present and within band
    y = (radius_ok & ins_ok).astype(int)
    # drop NA labels (if inputs missing)
    y[pd.isna(df["pl_rade"]) | pd.isna(df["pl_insol"])] = np.nan
    return y

def load_data():
    if not DATA_PARQUET.exists():
        raise FileNotFoundError("Run: python src/fetch_data.py first.")
    df = pd.read_parquet(DATA_PARQUET)
    return df

def main():
    df = load_data()

    # Target
    y = make_label(df)
    mask = ~y.isna()
    y = y[mask]

    # Features
    X = df.loc[mask, NUM_FEATURES + CAT_FEATURES].copy()

    # Train/test split (stratified)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    # Pipeline: impute numeric with median, categorical with "missing", RF classifier
    pre = ColumnTransformer(
        transformers=[
            ("num", SimpleImputer(strategy="median"), NUM_FEATURES),
            ("cat", Pipeline(steps=[
                ("imp", SimpleImputer(strategy="most_frequent")),
                ("oh", OneHotEncoder(handle_unknown="ignore"))
            ]), CAT_FEATURES),
        ],
        remainder="drop",
    )

    clf = RandomForestClassifier(
        n_estimators=400,
        max_depth=None,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )

    pipe = Pipeline(steps=[("pre", pre), ("clf", clf)])
    pipe.fit(X_train, y_train)

    # Metrics
    proba = pipe.predict_proba(X_test)[:, 1]
    pred = (proba >= 0.5).astype(int)

    roc_auc = roc_auc_score(y_test, proba)
    ap = average_precision_score(y_test, proba)  # PR-AUC
    f1 = f1_score(y_test, pred)
    cm = confusion_matrix(y_test, pred).tolist()
    cr = classification_report(y_test, pred, output_dict=True)

    metrics = {
        "roc_auc": float(roc_auc),
        "pr_auc": float(ap),
        "f1": float(f1),
        "confusion_matrix": cm,
        "classification_report": cr,
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "positive_rate_train": float(y_train.mean()),
        "positive_rate_test": float(y_test.mean()),
        "label_definition": {
            "insolation_range": HZ_BOUNDS["insolation"],
            "radius_range": HZ_BOUNDS["radius_Re"]
        },
        "features": {
            "numeric": NUM_FEATURES,
            "categorical": CAT_FEATURES
        }
    }

    joblib.dump(pipe, MODELS_DIR / "hz_rf.pkl")
    with open(MODELS_DIR / "hz_rf_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print("Saved model →", MODELS_DIR / "hz_rf.pkl")
    print("Saved metrics →", MODELS_DIR / "hz_rf_metrics.json")
    print(f"ROC-AUC: {roc_auc:.3f} | PR-AUC: {ap:.3f} | F1: {f1:.3f}")

if __name__ == "__main__":
    main()
