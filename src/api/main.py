from fastapi import FastAPI
from contextlib import asynccontextmanager

from src.models import load_models, predict
from src.explain import load_explainer, explain_transaction
from src.api.schemas import TransactionInput

MODELS = None
EXPLAINER = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global MODELS, EXPLAINER

    MODELS = load_models()
    EXPLAINER = load_explainer(MODELS["xgb"])

    yield

    # cleanup would go here if needed


app = FastAPI(lifespan=lifespan)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict_endpoint(input: TransactionInput):
    raw_input = input.model_dump()
    return predict(raw_input)


@app.post("/explain")
def explain_endpoint(input: TransactionInput):
    raw_input = input.model_dump()
    return explain_transaction(raw_input, MODELS, EXPLAINER)
