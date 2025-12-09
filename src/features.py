from __future__ import annotations


from pathlib import Path
from typing import Dict, Tuple, Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler,StandardScaler

from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest  # probably used later if you want
import joblib


def load_raw_data(path):
    path  = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    df = pd.read_csv(path) 
    return df 


def add_timestamp_features(df):

    df = df.copy()
    base_time = pd.to_datetime("2024-01-01 00:00:00")
    
    df["timestamp"] = base_time +  pd.to_timedelta(df['Time'], unit="s")
    df["hour"] = df["timestamp"].dt.hour.astype("int32")
    df["dayofweek"] = df["timestamp"].dt.dayofweek.astype("int32")

    return df


def add_amount_features(df, scaler=None):
    df = df.copy()

    # Log transform
    df["amount_log"] = np.log1p(df["Amount"])

    # Scaling
    if scaler is None:
        scaler = RobustScaler()
        df["amount_scaled"] = scaler.fit_transform(df[["Amount"]])
    else:
        df["amount_scaled"] = scaler.transform(df[["Amount"]])

    return df, scaler


def augment_synthetic_categories(df, seed=42):
    """
    Add synthetic merchant, device, geo, and account features.

    Adds:
        merchant_id
        device_type
        geo_bucket
        account_id
        account_age_days
    """
    df = df.copy()
    rng = np.random.default_rng(seed)

    n = len(df)

    # merchant_id

    n_merchants = 1000
    merchant_weights = rng.exponential(scale=1.0, size=n_merchants)
    merchant_weights = merchant_weights / merchant_weights.sum()
    df["merchant_id"] = rng.choice(n_merchants, size=n, p=merchant_weights)
    
    
    # device_type
    
    device_types = ["mobile", "desktop", "pos", "tablet"]
    device_probs = [0.60, 0.25, 0.10, 0.05]
    df["device_type"] = rng.choice(device_types, size=n, p=device_probs)

    # geo_bucket
    
    n_geo = 50
    geo_weights = rng.exponential(scale=1.0, size=n_geo)
    geo_weights = geo_weights / geo_weights.sum()
    df["geo_bucket"] = rng.choice(n_geo, size=n, p=geo_weights)

    # account_id
    n_accounts = 10000
    acct_weights = rng.exponential(scale=1.0, size=n_accounts)
    acct_weights = acct_weights / acct_weights.sum()
    df["account_id"] = rng.choice(n_accounts, size=n, p=acct_weights)
    
    # account_age_days

    account_ages = rng.integers(0, 2000, size=n_accounts)
    df["account_age_days"] = account_ages[df["account_id"]]

    return df


def add_frequency_features(df):
    """
    Add frequency-based features:
        merchant_freq
        device_freq
        account_txn_count
    """
    df = df.copy()

    # merchant frequency
    merchant_counts = df["merchant_id"].value_counts()
    df["merchant_freq"] = df["merchant_id"].map(merchant_counts)

    # device type frequency
    device_counts = df["device_type"].value_counts()
    df["device_freq"] = df["device_type"].map(device_counts)

    # account transaction count
    account_counts = df["account_id"].value_counts()
    df["account_txn_count"] = df["account_id"].map(account_counts)

    return df

def add_rolling_features(df):
    """
    Compute rolling behavioral features per account:
        - last_5_mean_amount
        - last_5_count
    """
    df = df.copy()

    # sort per account by timestamp
    df = df.sort_values(["account_id", "timestamp"])

    # rolling mean of last 5 amounts
    df["last_5_mean_amount"] = (
        df.groupby("account_id")["Amount"]
          .rolling(5)
          .mean()
          .shift()
          .reset_index(level=0, drop=True)
    ).fillna(0)

    # rolling count of last 5 transactions
    df["last_5_count"] = (
        df.groupby("account_id")["Amount"]
          .rolling(5)
          .count()
          .shift()
          .reset_index(level=0, drop=True)
    ).fillna(0)

    return df


def encode_categoricals(df, encoders=None):
    """
    Frequency-encode categorical variables.
    If encoders is None -> fit new encoding dicts.
    If encoders provided -> use existing mappings (inference mode).

    Returns:
        df, encoders
    """
    df = df.copy()

    cat_cols = ["merchant_id", "device_type", "geo_bucket", "account_id"]

    # Fit mode
    if encoders is None:
        encoders = {}
        for col in cat_cols:
            counts = df[col].value_counts()
            encoders[col] = counts.to_dict()       # store mapping
            df[col + "_fe"] = df[col].map(encoders[col]).fillna(0)
    else:
        # Transform mode
        for col in cat_cols:
            mapping = encoders[col]
            df[col + "_fe"] = df[col].map(mapping).fillna(0)

    return df, encoders


def add_interaction_features(df, new_merchant_threshold=50):
    """
    Add interaction-based fraud features:
        amount_times_age = Amount * account_age_days
        is_new_merchant  = 1 if merchant_freq < threshold else 0
    """
    df = df.copy()

    # Interaction term
    df["amount_times_age"] = df["Amount"] * df["account_age_days"]

    # New merchant indicator
    df["is_new_merchant"] = (df["merchant_freq"] < new_merchant_threshold).astype(int)

    return df

def add_missing_flags(df):
    """
    Add boolean flags indicating missing values for key categorical columns.
    """
    df = df.copy()

    target_cols = [
        "merchant_id",
        "device_type",
        "geo_bucket",
        "account_age_days"
    ]

    for col in target_cols:
        df[f"{col}_missing"] = df[col].isna().astype(int)

    return df

from sklearn.decomposition import PCA

def apply_pca(df, pca=None):
    """
    Apply PCA (2 components) for visualization.
    """
    df = df.copy()

    # Select numeric features only
    num_df = df.select_dtypes(include=["float64", "int64", "int32"])

    if pca is None:
        pca = PCA(n_components=2)
        comps = pca.fit_transform(num_df)
    else:
        comps = pca.transform(num_df)

    df["pca_x"] = comps[:, 0]
    df["pca_y"] = comps[:, 1]

    return df, pca


def feature_pipeline(df, fit=True, preprocessors=None, seed=42):
    """
    Master feature engineering pipeline.
    
    If fit=True:
        - fits scaler, encoders, PCA
    If fit=False:
        - uses provided preprocessors
    Returns:
        df_final, preprocessors_dict
    """
    df = df.copy()

    if fit:
        preprocessors = {}

    # 1. timestamp
    df = add_timestamp_features(df)

    # 2. amount features
    if fit:
        df, scaler = add_amount_features(df)
        preprocessors["scaler"] = scaler
    else:
        df, _ = add_amount_features(df, scaler=preprocessors["scaler"])

    # 3. synthetic categories
    df = augment_synthetic_categories(df, seed=seed)

    # 4. frequency features
    df = add_frequency_features(df)

    # 5. rolling features
    df = add_rolling_features(df)

    # 6. missing flags
    df = add_missing_flags(df)

    # 7. categorical encodings
    if fit:
        df, encoders = encode_categoricals(df)
        preprocessors["encoders"] = encoders
    else:
        df, _ = encode_categoricals(df, encoders=preprocessors["encoders"])

    # 8. interaction features
    df = add_interaction_features(df)

    # 9. PCA
    if fit:
        df, pca = apply_pca(df)
        preprocessors["pca"] = pca
    else:
        df, _ = apply_pca(df, pca=preprocessors["pca"])

    return df, preprocessors



def save_preprocessors(preprocessors, save_dir="../experiments/models"):
    """
    Save scaler, encoders, and PCA objects to disk using joblib.
    """
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    joblib.dump(preprocessors["scaler"], save_path / "scaler.joblib")
    joblib.dump(preprocessors["encoders"], save_path / "encoders.joblib")
    joblib.dump(preprocessors["pca"], save_path / "pca.joblib")

    print(f"Saved preprocessors to {save_path}")