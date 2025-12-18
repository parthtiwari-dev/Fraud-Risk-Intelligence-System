# **FRIS — Fraud Risk Intelligence System**
A production-grade fraud detection system designed to operate exactly where real-world banks and fintechs need it.

## **Why This Project? (Business Context)**
Financial institutions face three hard problems in fraud:

1. **Labels are delayed and incomplete** — you can’t rely purely on supervised learning.
2. **Inference must be deterministic** — regulators don’t accept "notebook logic".
3. **Every decision must be explainable** — internal audit, compliance, and customers demand clarity.

FRIS is built as the system that would sit **between transaction processing and risk decisioning** in a real bank/fintech:

- ingest a transaction
- compute engineered signals + model outputs
- combine supervised + unsupervised signals
- return a risk score and explanation
- forward the decision to fraud ops, rule engines, or customer flows

It acts as the **fraud intelligence layer**, not just a classifier.**
A full-stack, production-oriented fraud detection system. Built end-to-end with deterministic preprocessing, stacked ML inference, SHAP explainability, FastAPI serving, Docker packaging, and a Streamlit UI.

This repository contains the **complete system** exactly as it runs in production.

---

## 🚀 **Live Demo**
Below is the live version of the exact system described:
**
**Backend (FastAPI):** https://fraud-risk-intelligence-system-api.onrender.com/

**Frontend (Streamlit):** https://fraud-risk-intelligence-system.streamlit.app/

---

## 🧱 **System Architecture**

### High-Level Architecture Diagram
![System Architecture](assets/system_architecture.png)

### API Sequence Diagram
![API Flow](assets/api_sequence.png)

### Streamlit UI Screenshot
![UI Screenshot](assets/ui_screenshot.png)

---

Below is the text-form pipeline diagram:**
```
Raw Transaction (JSON)
        ↓
Input Validation (Pydantic)
        ↓
Frozen Feature Pipeline
        ↓
Base Models
  - XGBoost
  - Autoencoder (PyTorch)
  - Isolation Forest
  - MLP
        ↓
Meta-Feature Builder
        ↓
Stacked Ensemble (Logistic Regression)
        ↓
Risk Score + Label
        ↓
SHAP Explanation (Inference-Aligned)
        ↓
FastAPI → Streamlit UI
```

**One pipeline. One truth. Training = Inference = Explainability.**

---

## 🔍 **Tech Highlights**
These are the core engineering skills demonstrated by FRIS:

- **training/inference parity** (deterministic, frozen preprocessing)
- **stacked ensembles** with OOF predictions
- **SHAP explainability at inference** (no notebook drift)
- **FastAPI backend** with lifecycle-safe loading
- **Dockerized deployment** for reproducibility
- **CI-ready repository structure** with tests
- **strict schema validation** and boundary contracts
- **Streamlit frontend** correctly separated from ML logic

---

## 🔍 **Key Features****
### **1. Deterministic Feature Engineering**
All preprocessing is frozen:
- numerical transforms
- temporal features
- frequency encodings
- aggregation stats
- missingness flags

Stored as:
- `feature_columns.json`
- `preprocessors.joblib`

Inference never recomputes anything.

---

### **2. Hybrid Modeling**
FRIS combines multiple weak signals:
- **XGBoost** (supervised)
- **Autoencoder** (reconstruction-based anomaly signal)
- **Isolation Forest** (unsupervised)
- **MLP** (nonlinear auxiliary signal)

These feed into a **Logistic Regression stacker** trained only on **OOF predictions**.

---

### **3. Real Explainability (SHAP)**
FRIS implements:
- global importance
- local per-transaction attributions
- top-K feature drivers
- inference-aligned explanations

No notebook recomputation. No drift.

---

### **4. Production-Grade API**
FastAPI backend with:
- `GET /health`
- `POST /predict`
- `POST /explain`

Includes:
- lifecycle-safe model loading
- frozen contracts
- schema validation
- deterministic behavior across environments
- end-to-end tests

---

### **5. Deployment & Packaging**
- Dockerized backend
- artifacts baked in
- pinned requirements
- deployed on Render (API)
- deployed on Streamlit Cloud (UI)

---

### **6. Streamlit UI**
Minimal, narrative-style interface:
- enter a transaction
- get fraud probability + label
- see SHAP explanation
- transparency panel

UI contains **zero ML logic** — everything flows through the API.

---

## 📦 **Repository Structure**
```
FRIS/
├── src/
│   ├── api/               # FastAPI backend
│   ├── features/          # Frozen feature pipeline
│   ├── models/            # Model loaders + artifacts
│   ├── pipeline/          # Inference spine
│   ├── explain/           # SHAP explain layer
│   └── utils/             # Helpers
│
├── app/                   # Streamlit frontend
│   └── streamlit/
│
├── data/
│   ├── processed/
│   └── artifacts/         # models, encoders, scalers, explainer
│
├── tests/                 # API + inference tests
│
├── docs
├── notebooks
├── Dockerfile
├── requirements.txt
└── README.md
```

---

## 📡 **Usage**
### **Run backend locally:**
```
pip install -r requirements.txt
uvicorn src.api.main:app --reload
```

### **Call API:**
```
POST /predict
{
  "Time": 10000,
  "V1": -1.35,
  ...
  "Amount": 92.10
}

Response:
{
  "score": 0.87,
  "label": "fraud"
}
```

---

## 📊 **Dataset**
- 284,807 transactions
- 0.17% fraud rate
- PCA-derived features V1–V28
- Columns: Time, Amount, V1–V28, Class

Raw data not included.

---

## ✔️ **What FRIS Demonstrates**
- training/inference parity
- frozen preprocessing
- leakage detection
- stacked model design
- honest SHAP explainability
- API-first ML engineering
- Docker deployment
- UI separation of concerns
- real-world ML constraints

This is not a model.  
This is a **complete ML system**.

---

## 📌 **Project Status**
**FRIS v1.0 — Complete, deployed, stable.**

---

