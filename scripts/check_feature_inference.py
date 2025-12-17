"""
CHECKPOINT 1 — Feature Inference Sanity Check

Goal:
Verify that inference-time feature generation produces EXACTLY the same
engineered features as training-time feature generation.

This script tests ONLY the feature layer.
No models. No stacking. No SHAP.
"""

import pandas as pd
import numpy as np
from pathlib import Path

from src.features import feature_pipeline
from src.feature_inference import prepare_features

# ---------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "raw" / "creditcard.csv"        # adjust if needed
PREPROCESSORS_PATH = BASE_DIR / "experiments" / "models" / "preprocessors.joblib"

ROW_INDEX = 100   # any valid row index

# ---------------------------------------------------------------------
# LOAD RAW TRAINING DATA
# ---------------------------------------------------------------------

print("[1] Loading training data...")

df_raw = pd.read_csv(DATA_PATH)

raw_row = df_raw.iloc[ROW_INDEX].to_dict()

print(f"Using row index: {ROW_INDEX}")

# ---------------------------------------------------------------------
# TRAINING-TIME FEATURE GENERATION
# ---------------------------------------------------------------------

print("[2] Generating features using training pipeline...")

# NOTE:
# We assume preprocessors.joblib was saved during training
import joblib
preprocessors = joblib.load(PREPROCESSORS_PATH)


df_train_features, _ = feature_pipeline(
    pd.DataFrame([raw_row]),
    fit=False,
    preprocessors=preprocessors
)


# ---------------------------------------------------------------------
# INFERENCE-TIME FEATURE GENERATION
# ---------------------------------------------------------------------

print("[3] Generating features using inference pipeline...")

df_infer_features = prepare_features(raw_row)

# ---------------------------------------------------------------------
# COLUMN CHECK
# ---------------------------------------------------------------------

print("[4] Checking column equality...")

train_cols = set(df_train_features.columns)
infer_cols = set(df_infer_features.columns)

if train_cols != infer_cols:
    print("❌ Column mismatch detected")
    print("Missing in inference:", sorted(train_cols - infer_cols))
    print("Extra in inference:", sorted(infer_cols - train_cols))
    raise AssertionError("Feature columns do not match")

print("✅ Columns match")

# ---------------------------------------------------------------------
# VALUE CHECK (NUMERIC)
# ---------------------------------------------------------------------

print("[5] Checking numeric values...")

for col in train_cols:
    v_train = df_train_features[col].values
    v_infer = df_infer_features[col].values

    # datetime columns
    if np.issubdtype(v_train.dtype, np.datetime64):
        if not (v_train == v_infer).all():
            raise AssertionError(f"Datetime mismatch in column '{col}'")

    # numeric columns
    elif np.issubdtype(v_train.dtype, np.number):
        if not np.allclose(v_train, v_infer, rtol=1e-6, atol=1e-8):
            raise AssertionError(
                f"Numeric mismatch in column '{col}': "
                f"train={v_train[0]}, infer={v_infer[0]}"
            )

    # string / categorical columns
    else:
        if not (v_train == v_infer).all():
            raise AssertionError(
                f"Categorical mismatch in column '{col}': "
                f"train={v_train[0]}, infer={v_infer[0]}"
            )


print("✅ All feature values match")

# ---------------------------------------------------------------------
# FINAL VERDICT
# ---------------------------------------------------------------------

print("\n🎯 CHECKPOINT 1 PASSED")
print("Feature inference is IDENTICAL to training-time feature generation")
from src.features import save_preprocessors

save_preprocessors(
    preprocessors,
    save_dir="experiments/models"
)
