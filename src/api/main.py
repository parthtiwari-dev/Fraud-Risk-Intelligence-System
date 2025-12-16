from fastapi import FastAPI
from pydantic import BaseModel
from typing import Dict, Any

from src.models import load_models
from src.explain import load_explainer, explain_transaction


# -------------------------------
# App initialization
# -------------------------------

app = FastAPI(title="FRIS API")


# -------------------------------
# Input schema
# -------------------------------

class TransactionSchema(BaseModel):
    """
    Raw transaction input.
    This must match the input expected by validate_input().
    """
    data: Dict[str, Any]

    def to_dict(self):
        return self.data


# -------------------------------
# Startup: load models ONCE
# -------------------------------

@app.on_event("startup")
def load_artifacts():
    models = load_models()
    explainer = load_explainer(models["xgb"])

    app.state.models = models
    app.state.explainer = explainer


# -------------------------------
# Prediction + Explainability
# -------------------------------

@app.post("/predict")
def predict(input: TransactionSchema):
    models = app.state.models
    explainer = app.state.explainer

    # Raw input dict
    input_dict = input.to_dict()

    # Prediction
    score, label = models["predict"](input_dict)

    # Explainability
    top_features = explain_transaction(
        input_dict,
        models,
        explainer,
        k=5
    )

    return {
        "score": float(score),
        "label": int(label),
        "top_features": top_features
    }
