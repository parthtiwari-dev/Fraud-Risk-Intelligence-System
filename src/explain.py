"""
Explainability module for FRIS.
Computes SHAP values for single transactions using final XGB model.
"""

import shap
import pandas as pd

from .models import prepare_features



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
    Must follow EXACT same pipeline as predict().
    """

    # 1. Build engineered features using canonical pipeline
    df_eng = prepare_features(input_dict)


    # 2. Slice EXACT XGB feature list
    xgb_df = df_eng[models["xgb_features"]]

    # 3. Compute SHAP values
    shap_vals = compute_shap_single(explainer, xgb_df)

    # 4. Return top-k explanation
    return top_k_features(shap_vals, xgb_df, k)
