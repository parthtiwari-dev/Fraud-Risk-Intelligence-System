# FRIS — Fraud Risk Intelligence System

## What This Project Is

FRIS is a **production‑grade ML inference system**, not a notebook demo.

The goal was never just high metrics. The goal was to build a **correct, reproducible, deployable fraud detection system** that survives real‑world constraints:

- training ≠ inference
- features drift if not frozen
- models fail silently without contracts
- explainability must align with inference, not training

This repository represents the final, locked outcome of ~12 days of iterative ML + engineering work.

---

## Core Problem Solved

**How do you take a complex fraud ML pipeline and make it:**

- deterministic
- contract‑driven
- explainable
- deployable
- callable by external systems

without notebooks, hacks, or recomputation?

FRIS answers that.

---

## High‑Level Architecture

```
Raw Transaction (JSON)
        ↓
Input Validation (Schema)
        ↓
Frozen Feature Engineering Pipeline
        ↓
Base Model Inference
(XGBoost, RF, MLP, Anomaly models)
        ↓
Meta‑Feature Construction
        ↓
Stacked Ensemble (Logistic Regression)
        ↓
Thresholding (Cost‑aware)
        ↓
Prediction Output
        ↓
SHAP Explanation (Inference‑aligned)
```

One spine. One source of truth.

---

## Dataset

- Credit Card Fraud Dataset
- ~285k transactions
- ~0.17% fraud rate (extreme imbalance)
- PCA‑transformed features V1–V28
- Raw features: Time, Amount, Class

---

## EDA & Feature Engineering (Early Phase)

### EDA Performed

- Class imbalance analysis
- Fraud vs non‑fraud distributions
- Amount statistics (min, max, log‑scale)
- Time monotonicity checks
- Hour‑of‑day and day‑of‑week analysis
- Correlation heatmaps
- PCA scatter visualization (V1 vs V2)
- Z‑score anomaly analysis
- Duplicate and missing value checks

Key finding:
- Dataset is clean
- Fraud signals are subtle
- Raw PCA features alone are insufficient

---

### Engineered Features (Frozen)

**Numerical / Transformations**
- amount_log
- amount_scaled
- amount_times_age

**Temporal**
- hour
- dayofweek
- account_age_days

**Frequency / Aggregates**
- merchant_freq
- account_txn_count
- device_freq
- last_5_mean_amount
- last_5_count

**Categorical (Encoded)**
- merchant_id_fe
- device_type_fe
- geo_bucket_fe
- account_id_fe

**Missingness Flags**
- merchant_id_missing
- device_type_missing
- geo_bucket_missing
- account_age_days_missing

These features are frozen via:
- `features.py`
- `feature_columns.json`
- `preprocessors.joblib`

---

## Modeling Pipeline

### Baselines

1. Logistic Regression
   - class_weight = balanced
   - high recall, very low precision

2. Random Forest
   - better precision
   - weaker recall

---

### XGBoost (Primary Model)

- Stratified K‑Fold CV
- PR‑AUC as primary metric
- Grid search over:
  - max_depth
  - learning_rate
  - subsample
  - colsample_bytree
  - n_estimators

Best tuned model selected and retrained on full data.

---

### Threshold Optimization

Instead of default 0.5:

- Manual threshold sweep (0.01–0.99)
- Explicit cost function:
  - FN cost >> FP cost

Final threshold chosen based on **minimum expected cost**, not F1 alone.

Threshold is frozen and logged.

---

## Unsupervised & Neural Models

### Anomaly Detection

- Isolation Forest
- PCA‑based anomaly score
- Used as **signals**, not decisions

### Autoencoder (PyTorch)

- Encoder: 55 → 128 → 64 → 32 → 8
- Reconstruction error used as signal
- Latent features explored

---

### MLP

- Trained on engineered features
- Used as base learner
- Outputs probability only

---

## Stacked Ensemble (Critical Phase)

### Base Signals

- xgb_proba
- rf_proba
- mlp_proba
- anomaly_score
- ae_recon_error
- cluster_id

### Initial Failure

- Near‑perfect CV metrics
- PR‑AUC ≈ 1.0

Root cause:
**Leakage from using full‑train predictions inside CV.**

---

### Corrected Stacking

- Strict out‑of‑fold predictions
- Meta‑features built ONLY from OOF outputs
- Logistic Regression as stacker

Final honest metrics:
- PR‑AUC ≈ 0.81
- Recall ≈ 0.86

---

## Training vs Inference Crisis (Major Learning)

Multiple failures occurred:

- Feature count mismatches
- Missing meta‑features at inference
- PCA columns leaking into stacker
- Synthetic categorical regeneration at inference

Final resolution:

**Models do not know features. Dataframes do.**

Solution:
- Explicit feature lists per model
- Frozen slicing logic
- No guessing at inference

---

## Inference Architecture (Locked)

- Raw JSON only
- No DataFrame assumptions
- One canonical pipeline

Key rules:
- Feature engineering runs ONCE
- Models load ONCE
- No recomputation
- No notebooks in inference

---

## Explainability (SHAP)

### Global
- SHAP summary plots
- Top feature importance

### Local
- Transaction‑level SHAP
- Top‑K contributing features
- Aligned with inference pipeline

Important:
SHAP never modifies data or features.

---

## API Layer (FastAPI)

Endpoints:
- `/health`
- `/predict`
- `/explain`

Characteristics:
- Lifecycle‑safe loading
- Models and explainers loaded once
- Pydantic schemas enforce contracts
- Identical behavior in local, tests, prod

---

## Testing

- FastAPI TestClient
- End‑to‑end inference tests
- Schema validation tests
- Feature contract verification scripts

---

## Dockerization

- Python 3.11 pinned
- Training and inference environments matched
- All artifacts baked into image
- No runtime downloads

Key lesson:
Docker is packaging, not hosting.

---

## Deployment

- Backend deployed on Render
- Public HTTPS API
- Swagger available at `/docs`

Frontend:
- Streamlit UI
- Calls API only
- Zero ML logic

---

## What This Project Demonstrates

- Real ML systems thinking
- Training vs inference discipline
- Feature contracts
- Leakage detection
- Cost‑aware evaluation
- Explainable AI
- API‑first deployment

This is not a model showcase.
It is an **engineering system**.

---

## Project Status

**FRIS v1.0 — CLOSED**

- ML frozen
- API frozen
- Artifacts trusted
- Deployment live

Anything beyond this is a new project.

