# 🚨 FRIS API — Fraud Risk Intelligence System

This document explains **everything built in the API layer** of FRIS.
Not theory. Not hype. Actual engineering decisions, structure, and behavior.

If you read this end‑to‑end, you should understand **how a real ML model is served in production**.

---

## 🎯 What This API Does

FRIS exposes a **production‑grade ML inference + explainability service** over HTTP.

It allows you to:

- 🧠 Send a **raw transaction** (no engineered features)
- ⚡ Get a **fraud risk score + label**
- 🔍 Get **SHAP‑based explanations** aligned with the prediction

All of this happens:
- deterministically
- using frozen feature contracts
- with identical training vs inference behavior

---

## 🧱 High‑Level Architecture

```
Client (Swagger / Script / Frontend)
        ↓ JSON
FastAPI (API layer)
        ↓
Feature Inference (frozen pipeline)
        ↓
Models (XGB + Stack)
        ↓
Prediction / SHAP
        ↓
HTTP JSON Response
```

Key principle:
> **The API does NOT do ML. It only wires requests to already‑verified logic.**

---

## 📁 API Folder Structure

```
src/api/
├── main.py        # FastAPI app + lifecycle + routes
├── schemas.py     # Input contracts (Pydantic)
├── test_client.py # End‑to‑end API tests
└── __init__.py
```

Each file has a single responsibility. No overlap. No magic.

---

## 🚀 `main.py` — The API Entry Point

### Responsibilities

- Create the FastAPI app
- Load ML artifacts **once at startup**
- Expose HTTP endpoints
- Never contain ML logic

### Lifespan (Startup Logic)

We use FastAPI **lifespan** instead of deprecated startup hooks:

- Models are loaded once
- SHAP explainer is built once
- Objects live for the entire app lifetime

This guarantees:
- ⚡ No reload per request
- 🧠 Consistent inference
- 🧪 Tests behave like production

---

## 🔌 Global Objects (Why They Exist)

```python
MODELS = None
EXPLAINER = None
```

These are:
- read‑only
- initialized at startup
- reused across requests

This is **standard practice in ML serving**.

---

## 🛣️ API Endpoints

### 🩺 `GET /health`

Purpose:
- Check service liveness
- Used by load balancers, Docker, Kubernetes

Response:
```json
{ "status": "ok" }
```

---

### ⚡ `POST /predict`

Purpose:
- Run fraud inference

Input:
- Raw transaction JSON
- Validated by schema

Output:
```json
{
  "score": 0.0,
  "label": "legit"
}
```

Rules:
- No feature engineering here
- No SHAP here
- Fast and deterministic

---

### 🔍 `POST /explain`

Purpose:
- Explain **exactly the same prediction**

Output:
```json
[
  { "feature": "V3", "value": 2.53, "shap_value": -0.86 },
  ...
]
```

Rules:
- Uses same feature row as prediction
- No recomputation
- No guessing columns

---

## 📜 `schemas.py` — API Contracts

Schemas define **what the API accepts**, not what the model uses internally.

### `TransactionInput`

- Matches raw dataset exactly
- Includes:
  - `Time`
  - `V1` … `V28`
  - `Amount`
- ❌ No engineered features
- ❌ No `Class`

Benefits:
- 🧱 Strong boundary validation
- 📖 Auto‑generated docs
- 🧠 No garbage reaches the model

---

## 🧪 `test_client.py` — Why This Exists

These are **true end‑to‑end tests**.

They verify:
- API starts correctly
- Lifespan runs
- Models load
- `/health` works
- `/predict` returns valid structure
- `/explain` returns SHAP output

Important detail:

```python
with TestClient(app) as client:
```

This ensures:
- startup lifecycle is executed
- tests mirror real production behavior

Without this, tests lie.

---

## 🧠 Key Engineering Lessons

### ✅ Contracts First

- Feature columns frozen
- Schema enforced
- No silent mismatches

### ✅ Training = Inference

- Same pipeline
- Same columns
- Same order

### ✅ Boring is Good

- No refactors
- No clever async
- No hidden state

### ✅ Loud Failures > Silent Bugs

- Validation errors are good
- Startup crashes are good
- Early failure prevents production disasters

---

## 🧑‍💻 How to Run Locally

```bash
uvicorn src.api.main:app --reload
```

Open:
- 📖 Docs: http://127.0.0.1:8000/docs
- 🩺 Health: http://127.0.0.1:8000/health

---

## 🧪 How to Test

```bash
pytest src/api/test_client.py
```

All tests must pass before:
- Docker
- Deployment
- Refactors

---

## 🚫 Things We Explicitly Avoided

- Re‑engineering features
- Wrapping models in magic classes
- Combining predict + explain prematurely
- Async complexity
- Hidden state mutation

These mistakes break ML systems quietly.

---

## 🏁 Final State

At this point, FRIS has:

- ✅ Production‑grade ML API
- ✅ Deterministic inference
- ✅ Explainability
- ✅ Schema‑validated inputs
- ✅ Lifecycle‑safe startup
- ✅ End‑to‑end tests

This API can now be:
- Dockerized
- Deployed
- Used by any client

---

🔥 **This is not a tutorial API. This is how real ML systems are shipped.**

