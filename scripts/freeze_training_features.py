"""
STEP 2.5 — Freeze Training Feature Contract

Purpose:
- Run the feature engineering pipeline ONCE in training mode
- Freeze everything inference depends on

This script does NOT train models.
This script does NOT evaluate models.
This script ONLY freezes feature-related artifacts.

Run this exactly once after you are confident your feature pipeline is correct.
"""

import json
from pathlib import Path
import pandas as pd
import joblib

from src.features import feature_pipeline

 
# CONFIG — adjust paths if needed
 

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "raw"/ "creditcard.csv"          # raw training data
SAVE_DIR = BASE_DIR / "experiments" / "models"      # where artifacts go

SAVE_DIR.mkdir(parents=True, exist_ok=True)

PREPROCESSORS_PATH = SAVE_DIR / "preprocessors.joblib"
FEATURE_COLUMNS_PATH = SAVE_DIR / "feature_columns.json"

 
# LOAD RAW TRAINING DATA
 

print("[1] Loading raw training data...")

df_train = pd.read_csv(DATA_PATH)

print(f"Loaded dataset with shape: {df_train.shape}")

 
# RUN FEATURE PIPELINE IN TRAINING MODE
 

print("[2] Running feature pipeline in FIT mode...")

df_train_features, preprocessors = feature_pipeline(
    df_train,
    fit=True
)

print(f"Engineered feature shape: {df_train_features.shape}")

 
# SAVE PREPROCESSORS
 

print("[3] Saving preprocessors...")

joblib.dump(preprocessors, PREPROCESSORS_PATH)

print(f"Saved preprocessors to: {PREPROCESSORS_PATH}")

 
# SAVE FEATURE COLUMN CONTRACT
 

print("[4] Saving feature column contract...")

feature_columns = df_train_features.columns.tolist()

with open(FEATURE_COLUMNS_PATH, "w") as f:
    json.dump({"features": feature_columns}, f, indent=2)

print(f"Saved feature columns to: {FEATURE_COLUMNS_PATH}")

 
# FINAL VERDICT
 

print("\n✅ TRAINING FEATURE FREEZE COMPLETE")
print("You can now safely run inference and sanity checks.")
