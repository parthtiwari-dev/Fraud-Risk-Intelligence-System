"""
Explainability module for FRIS.
Computes SHAP values for single transactions using final XGB model.
"""

import shap
import pandas as pd

from .models import validate_input


def load_explainer(xgb_model):
    """
    Initialize SHAP TreeExplainer for the final XGBoost model.
    Call ONCE at API startup.
    """
    return shap.TreeExplainer(xgb_model)


def compute_shap_single(explainer, xgb_df):
    """
    Compute SHAP values for a single XGB-ready row.
    """
    shap_values = explainer.shap_values(xgb_df)
    return shap_values[0]


def top_k_features(shap_values, xgb_df, k=5):
    """
    Return top-k SHAP contributors in JSON-safe format.
    """
    feature_names = xgb_df.columns.tolist()

    pairs = list(zip(feature_names, shap_values))
    pairs_sorted = sorted(pairs, key=lambda x: abs(x[1]), reverse=True)

    return [
        {
            "feature": name,
            "shap_value": float(value),
            "value": float(xgb_df[name].iloc[0]),
        }
        for name, value in pairs_sorted[:k]
    ]


def explain_transaction(input_dict, models, explainer, k=5):
    """
    Explain XGBoost prediction for a single transaction.
    """

    # 1️⃣ Build FULL engineered feature DataFrame
    full_df = validate_input(input_dict)
    # full_df columns == xgb_features + others

    # 2️⃣ Slice EXACT XGB features
    xgb_df = full_df[models["xgb_features"]]

    # 3️⃣ Compute SHAP
    shap_vals = compute_shap_single(explainer, xgb_df)

    # 4️⃣ Return top-k explanation
    return top_k_features(shap_vals, xgb_df, k)
