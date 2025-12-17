"""
Feature inference layer for FRIS.

Responsibility:
- Convert ONE raw transaction dict into the EXACT engineered feature DataFrame
  used during training.

This file contains NO model logic, NO explainability logic, and NO shortcuts.
It is the single legal entry point for inference-time feature creation.
"""

from pathlib import Path
import json
import pandas as pd
import joblib

from .features import feature_pipeline

 
# Paths
 

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "experiments" / "models"

PREPROCESSORS_PATH = MODEL_DIR / "preprocessors.joblib"
FEATURE_COLUMNS_PATH = MODEL_DIR / "feature_columns.json"


 
# Helpers
 

def _load_preprocessors():
    """
    Load preprocessing objects fitted during training.
    """
    if not PREPROCESSORS_PATH.exists():
        raise FileNotFoundError(f"Preprocessors not found at {PREPROCESSORS_PATH}")

    return joblib.load(PREPROCESSORS_PATH)


def _load_training_feature_columns():
    """
    Load the exact feature column contract from training time.
    """
    if not FEATURE_COLUMNS_PATH.exists():
        raise FileNotFoundError(
            f"Feature column contract not found at {FEATURE_COLUMNS_PATH}"
        )

    with open(FEATURE_COLUMNS_PATH, "r") as f:
        data = json.load(f)

    if "features" not in data or not isinstance(data["features"], list):
        raise ValueError("Invalid feature_columns.json format")

    return set(data["features"])


 
# Public API
 

def prepare_features(raw_input: dict) -> pd.DataFrame:
    """
    Convert a single raw transaction dict into a fully engineered DataFrame.

    Parameters
    ----------
    raw_input : dict
        Raw transaction input representing ONE transaction.

    Returns
    -------
    pd.DataFrame
        Engineered feature DataFrame of shape (1, N).
    """

     
    # 1. Raw dict -> single-row DataFrame
     

    if not isinstance(raw_input, dict):
        raise TypeError("raw_input must be a dict")

    df_raw = pd.DataFrame([raw_input])

    if len(df_raw) != 1:
        raise ValueError("prepare_features supports exactly ONE transaction")

     
    # 2. Load preprocessors
     

    preprocessors = _load_preprocessors()

     
    # 3. Run feature engineering in inference mode
     

    df_eng, _ = feature_pipeline(
        df_raw,
        fit=False,
        preprocessors=preprocessors
    )

     
    # 4. Enforce training-time feature contract
     

    expected_features = _load_training_feature_columns()
    produced_features = set(df_eng.columns)

    missing = expected_features - produced_features
    extra = produced_features - expected_features

    if missing or extra:
        raise ValueError(
            "Feature contract mismatch. "
            f"Missing: {sorted(missing)} | Extra: {sorted(extra)}"
        )

     
    # 5. Return engineered DataFrame
     

    return df_eng
