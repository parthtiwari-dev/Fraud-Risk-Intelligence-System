"""
FRIS — Models Inference Layer (REFRACTORED)

Responsibilities:
- Consume ENGINEERED features only
- Generate base model signals
- Build meta-features
- Run stacker
- Return final score + label

This file DOES NOT:
- Engineer features
- Validate raw input
- Touch SHAP
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from joblib import load
import torch
import torch.nn as nn

from src.feature_inference import prepare_features

# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "experiments" / "models"
ENSEMBLE_DIR = BASE_DIR / "experiments" / "ensemble"

METRICS_PATH = ENSEMBLE_DIR / "metrics.json"

XGB_FEATURES_PATH = MODEL_DIR / "xgb_features.json"
IFOREST_FEATURES_PATH = MODEL_DIR / "iforest_features.json"
AE_FEATURES_PATH = MODEL_DIR / "ae_features.json"

# ---------------------------------------------------------------------
# Global cache
# ---------------------------------------------------------------------

_MODELS = None

# ---------------------------------------------------------------------
# Autoencoder definition (for loading weights)
# ---------------------------------------------------------------------

class Autoencoder(nn.Module):
    def __init__(self, input_dim):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 8),
        )

        self.decoder = nn.Sequential(
            nn.Linear(8, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, input_dim),
        )

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)

# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def _load_feature_list(path: Path, name: str) -> list:
    if not path.exists():
        raise FileNotFoundError(f"{name} feature file not found: {path}")

    with open(path, "r") as f:
        data = json.load(f)

    if "features" not in data or not isinstance(data["features"], list):
        raise ValueError(f"Invalid feature file format: {path}")

    return data["features"]

# ---------------------------------------------------------------------
# Load all frozen artifacts
# ---------------------------------------------------------------------

def load_models():
    global _MODELS

    if _MODELS is not None:
        return _MODELS

    if not METRICS_PATH.exists():
        raise FileNotFoundError(f"metrics.json not found at {METRICS_PATH}")

    with open(METRICS_PATH, "r") as f:
        metrics = json.load(f)

    meta_features = metrics["meta_features"]
    threshold = metrics["final_threshold"]

    xgb_features = _load_feature_list(XGB_FEATURES_PATH, "XGBoost")
    iforest_features = _load_feature_list(IFOREST_FEATURES_PATH, "IsolationForest")
    ae_features = _load_feature_list(AE_FEATURES_PATH, "Autoencoder")

    stacker = load(MODEL_DIR / "stacker.joblib")
    xgb = load(MODEL_DIR / "xgb.joblib")
    iforest = load(MODEL_DIR / "iforest.joblib")

    ae_state = torch.load(MODEL_DIR / "autoencoder.pt", map_location="cpu")
    ae_input_dim = ae_state["encoder.0.weight"].shape[1]

    autoencoder = Autoencoder(ae_input_dim)
    autoencoder.load_state_dict(ae_state)
    autoencoder.eval()

    _MODELS = {
        "stacker": stacker,
        "xgb": xgb,
        "iforest": iforest,
        "autoencoder": autoencoder,
        "xgb_features": xgb_features,
        "iforest_features": iforest_features,
        "ae_features": ae_features,
        "meta_features": meta_features,
        "threshold": threshold,
    }

    return _MODELS

# ---------------------------------------------------------------------
# Base signal generation
# ---------------------------------------------------------------------

def compute_base_signals(df_eng: pd.DataFrame) -> dict:
    if df_eng.shape[0] != 1:
        raise ValueError("compute_base_signals expects a single-row DataFrame")

    models = load_models()

    # XGBoost
    X_xgb = df_eng[models["xgb_features"]].values
    xgb_proba = float(models["xgb"].predict_proba(X_xgb, validate_features=False)[0, 1])

    # IsolationForest
    X_if = df_eng[models["iforest_features"]].values
    anomaly_score = float(-models["iforest"].decision_function(X_if)[0])

    # Autoencoder reconstruction error
    X_ae = torch.from_numpy(df_eng[models["ae_features"]].values.astype(np.float32))
    with torch.no_grad():
        recon = models["autoencoder"](X_ae)
        ae_recon_error = torch.mean((recon - X_ae) ** 2, dim=1).item()

    return {
        "xgb_proba": xgb_proba,
        "anomaly_score": anomaly_score,
        "ae_recon_error": ae_recon_error,
    }

# ---------------------------------------------------------------------
# Meta-feature construction
# ---------------------------------------------------------------------

def build_meta_features(df_eng: pd.DataFrame, base_signals: dict) -> np.ndarray:
    models = load_models()

    meta_row = {
        "xgb_oof_proba": base_signals["xgb_proba"],
        "anomaly_score": base_signals["anomaly_score"],
        "ae_recon_error": base_signals["ae_recon_error"],
        "amount_log": float(df_eng.iloc[0]["amount_log"]),
        "merchant_freq": float(df_eng.iloc[0]["merchant_freq"]),
        "account_txn_count": float(df_eng.iloc[0]["account_txn_count"]),
        "last_5_mean_amount": float(df_eng.iloc[0]["last_5_mean_amount"]),
    }

    meta_features = models["meta_features"]

    missing = [f for f in meta_features if f not in meta_row]
    if missing:
        raise ValueError(f"Missing meta features: {missing}")

    X_meta = np.array([[meta_row[f] for f in meta_features]], dtype=np.float32)
    return X_meta

# ---------------------------------------------------------------------
# Final prediction API
# ---------------------------------------------------------------------

def predict(raw_input: dict) -> dict:
    models = load_models()

    # 1. Raw -> engineered features
    df_eng = prepare_features(raw_input)

    # 2. Base model signals
    base_signals = compute_base_signals(df_eng)

    # 3. Meta-feature vector
    X_meta = build_meta_features(df_eng, base_signals)

    # 4. Stacker
    score = float(models["stacker"].predict_proba(X_meta)[0, 1])

    # 5. Threshold
    label = "fraud" if score >= models["threshold"] else "legit"

    return {
        "score": score,
        "label": label
    }
