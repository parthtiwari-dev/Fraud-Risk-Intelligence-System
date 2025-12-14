# src/models.py

import json
import numpy as np
import pandas as pd
from pathlib import Path
from joblib import load
import torch
import torch.nn as nn


# ============================================================
# Paths
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_DIR = BASE_DIR / "experiments" / "models"
ENSEMBLE_DIR = BASE_DIR / "experiments" / "ensemble"

METRICS_PATH = ENSEMBLE_DIR / "metrics.json"

XGB_FEATURES_PATH = MODEL_DIR / "xgb_features.json"
IFOREST_FEATURES_PATH = MODEL_DIR / "iforest_features.json"
AE_FEATURES_PATH = MODEL_DIR / "ae_features.json"


# ============================================================
# Global cache
# ============================================================

_MODELS = None


# ============================================================
# Autoencoder definition (needed for weight loading)
# ============================================================

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


# ============================================================
# Helpers
# ============================================================

def _load_feature_list(path: Path, name: str) -> list:
    if not path.exists():
        raise FileNotFoundError(f"{name} feature file not found: {path}")

    with open(path, "r") as f:
        data = json.load(f)

    if "features" not in data or not isinstance(data["features"], list):
        raise ValueError(f"Invalid feature file format: {path}")

    return data["features"]


# ============================================================
# Load all frozen artifacts
# ============================================================

def load_models():
    """
    Load and cache all inference-time artifacts.
    """
    global _MODELS

    if _MODELS is not None:
        return _MODELS

    # ---------------------------
    # metrics.json
    # ---------------------------
    if not METRICS_PATH.exists():
        raise FileNotFoundError(f"metrics.json not found at {METRICS_PATH}")

    with open(METRICS_PATH, "r") as f:
        metrics = json.load(f)

    meta_features = metrics["meta_features"]
    final_threshold = metrics["final_threshold"]

    # ---------------------------
    # Feature contracts
    # ---------------------------
    xgb_features = _load_feature_list(XGB_FEATURES_PATH, "XGBoost")
    iforest_features = _load_feature_list(IFOREST_FEATURES_PATH, "IsolationForest")
    ae_features = _load_feature_list(AE_FEATURES_PATH, "Autoencoder")
    
    required_input_features = sorted(
        set(xgb_features) |
        set(iforest_features) |
        set(ae_features)
    )

    # ---------------------------
    # Models
    # ---------------------------
    stacker = load(MODEL_DIR / "stacker.joblib")
    xgb = load(MODEL_DIR / "xgb.joblib")
    iforest = load(MODEL_DIR / "iforest.joblib")

    # Autoencoder
    ae_state = torch.load(MODEL_DIR / "autoencoder.pt", map_location="cpu")
    ae_input_dim = ae_state["encoder.0.weight"].shape[1]

    autoencoder = Autoencoder(ae_input_dim)
    autoencoder.load_state_dict(ae_state)
    autoencoder.eval()

    # ---------------------------
    # Cache
    # ---------------------------
    _MODELS = {
        "stacker": stacker,
        "xgb": xgb,
        "iforest": iforest,
        "autoencoder": autoencoder,
        "xgb_features": xgb_features,
        "iforest_features": iforest_features,
        "ae_features": ae_features,
        "meta_features": meta_features,
        "threshold": final_threshold,
        "required_input_features": required_input_features,
    }

    return _MODELS 


# ============================================================
# Input validation
# ============================================================

def validate_input(input_dict: dict) -> pd.DataFrame:
    """
    Validate FULL engineered feature vector.
    """
    if not isinstance(input_dict, dict):
        raise TypeError("Input must be a dict")

    models = load_models()
    required_features = models["required_input_features"]

    missing = [f for f in required_features if f not in input_dict]
    if missing:
        raise ValueError(f"Missing required features: {missing}")

    data = {}
    for f in required_features:
        v = input_dict[f]
        if not isinstance(v, (int, float, np.number)):
            raise TypeError(f"Feature '{f}' must be numeric, got {type(v)}")
        data[f] = float(v)

    return pd.DataFrame([data])


# ============================================================
# Base signal generation
# ============================================================

def compute_base_signals(input_df: pd.DataFrame) -> dict:
    """
    Generate base model signals from engineered features.
    """
    if input_df.shape[0] != 1:
        raise ValueError("compute_base_signals expects a single-row DataFrame")

    models = load_models()

    # ---------------------------
    # XGBoost
    # ---------------------------
    X_xgb = input_df[models["xgb_features"]].values
    xgb_proba = float(
        models["xgb"].predict_proba(
            X_xgb,
            validate_features=False
        )[0, 1]
    )


    # ---------------------------
    # IsolationForest
    # Higher = more anomalous
    # ---------------------------
    X_if = input_df[models["iforest_features"]].values
    anomaly_score = float(-models["iforest"].decision_function(X_if)[0])

    # ---------------------------
    # Autoencoder reconstruction error
    # ---------------------------
    X_ae = input_df[models["ae_features"]].values.astype(np.float32)
    X_ae = torch.from_numpy(X_ae)

    with torch.no_grad():
        recon = models["autoencoder"](X_ae)
        ae_recon_error = torch.mean((recon - X_ae) ** 2, dim=1).item()

    return {
        "xgb_proba": xgb_proba,
        "anomaly_score": anomaly_score,
        "ae_recon_error": ae_recon_error,
    }


def build_meta_features(input_dict: dict) -> np.ndarray:
    """
    Build the exact meta-feature vector expected by the stacker.
    Returns: np.ndarray of shape (1, n_meta_features)
    """
    # --------------------------------------------------
    # 1. Validate full engineered input
    # --------------------------------------------------
    df = validate_input(input_dict)

    # --------------------------------------------------
    # 2. Compute base model signals
    # --------------------------------------------------
    base_signals = compute_base_signals(df)

    # --------------------------------------------------
    # 3. Assemble meta-feature dict
    # --------------------------------------------------
    meta_row = {}

    # base signals
    meta_row["xgb_oof_proba"] = base_signals["xgb_proba"]  # OOF → final model at inference
    meta_row["anomaly_score"] = base_signals["anomaly_score"]
    meta_row["ae_recon_error"] = base_signals["ae_recon_error"]

    # cluster id comes from engineered input
    meta_row["cluster_id"] = float(df.iloc[0]["cluster_id"])


    # contextual raw features (already validated)
    meta_row["amount_log"] = float(df.iloc[0]["amount_log"])
    meta_row["merchant_freq"] = float(df.iloc[0]["merchant_freq"])
    meta_row["account_txn_count"] = float(df.iloc[0]["account_txn_count"])
    meta_row["last_5_mean_amount"] = float(df.iloc[0]["last_5_mean_amount"])

    # --------------------------------------------------
    # 4. Order features EXACTLY as stacker expects
    # --------------------------------------------------
    models = load_models()
    meta_features = models["meta_features"]

    missing = [f for f in meta_features if f not in meta_row]
    if missing:
        raise ValueError(f"Missing meta features: {missing}")

    X_meta = np.array(
        [[meta_row[f] for f in meta_features]],
        dtype=np.float32
    )

    return X_meta

def predict_proba(input_dict: dict) -> float:
    """
    Return fraud probability from the stacking model.
    """
    models = load_models()

    X_meta = build_meta_features(input_dict)
    proba = float(models["stacker"].predict_proba(X_meta)[0, 1])

    return proba


def predict(input_dict: dict) -> dict:
    """
    Final fraud prediction with thresholding.
    """
    models = load_models()

    score = predict_proba(input_dict)
    threshold = models["threshold"]

    label = "fraud" if score >= threshold else "legit"

    return {
        "score": score,
        "label": label
    }
