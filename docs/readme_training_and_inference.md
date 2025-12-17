# FRIS — Training & Inference README (READ THIS BEFORE TOUCHING ANYTHING)

This document exists because the project hit multiple **feature-contract and training–inference mismatches** during development.

If you skip this file and start editing notebooks or models blindly, you WILL break the system again.

This README explains:
- what was trained
- what went wrong
- why it happened
- what is frozen now
- how to safely move forward

---

## 1. High-level architecture (final truth)

FRIS is a **stacked fraud detection system**.

Base models:
- XGBoost (supervised)
- IsolationForest (unsupervised)
- Autoencoder (unsupervised)

Meta model:
- Logistic Regression stacker

Key principle:
> Training and inference MUST see the same feature contracts.

---

## 2. Feature engineering pipeline (what actually happened)

All feature engineering is done in `features.py` and notebooks.

The engineered dataset eventually contained:
- raw transaction features (Time, V1–V28, Amount)
- temporal features (hour, dayofweek)
- behavioral features (merchant_freq, account_txn_count, rolling stats)
- categorical encodings (target/frequency encoded)
- PCA outputs (pca_x, pca_y)
- IsolationForest outputs (anomaly_score, is_anomaly)
- clustering output (cluster_id)

Important:
- Feature selection was **implicit**, not explicit
- Any column present at scaling time became part of model training

This caused later confusion.

---

## 3. What went wrong (root causes)

### 3.1 Hidden feature drift
Models were trained on DataFrames where feature inclusion was implicit.

This led to:
- XGBoost being trained on unintended columns
- Autoencoder being trained on PCA, anomaly scores, and cluster IDs
- Inference code assuming a smaller feature set

Result:
- shape mismatches
- missing columns
- silent wrong assumptions

---

### 3.2 OOF vs inference confusion
During training:
- `xgb_oof_proba` was used correctly for stacking to avoid leakage

During inference:
- OOF does not exist
- Final XGB probability must be used instead

This required an **explicit mapping** at inference time.

---

### 3.3 Jupyter notebook side effects
Jupyter hid:
- exact feature order
- exact feature inclusion
- exact scaling inputs

This made debugging significantly harder.

Lesson:
> Notebooks hide contracts. Production code cannot.

---

## 4. What is frozen now (DO NOT CHANGE)

The following are **locked contracts**.

### 4.1 Base model feature contracts

Each base model expects the exact features it was trained on.

These are stored explicitly as JSON:
- `xgb_features.json`
- `iforest_features.json`
- `ae_features.json`

Inference slices input strictly using these lists.

---

### 4.2 Meta-feature contract (stacker)

Defined in `metrics.json`:

meta_features = [
"xgb_oof_proba",
"anomaly_score",
"ae_recon_error",
"cluster_id",
"amount_log",
"merchant_freq",
"account_txn_count",
"last_5_mean_amount"
]


Important mapping at inference:
- `xgb_oof_proba` ← final `xgb_proba`
- `cluster_id` ← engineered input

Order is critical. Do not reorder.

---

### 4.3 Inference responsibilities

`src/models.py`:
- does NOT do feature engineering
- does NOT do training
- does NOT recompute PCA, clustering, or encoders
- only loads artifacts and runs inference deterministically

---

## 5. Autoencoder note (technical debt)

The autoencoder was trained on:
- base engineered features
- PLUS PCA outputs
- PLUS IsolationForest outputs
- PLUS cluster_id

This is architecturally impure.

However:
- The stacker learned to downweight noise
- The system works end-to-end
- Retraining now would require rebuilding the entire stack

Decision:
> Accept this for v1. Refactor later if needed.

This is documented debt, not a hidden bug.

---

## 6. Why inference now works

Inference now:
- enforces full engineered input via `required_input_features`
- slices per-model features explicitly
- maps training-time constructs to inference-time equivalents
- guarantees correct ordering before predictions

If inference breaks again, it means:
- a feature list was changed
- a model artifact was retrained
- or this README was ignored

---

## 7. Rules for future changes (NON-NEGOTIABLE)

Before retraining ANY model:
1. Explicitly define feature lists
2. Save them to JSON
3. Train using only those columns
4. Update inference slicing accordingly

If you retrain:
- AE → must retrain stacker
- XGB → must retrain stacker
- Feature set → must retrain everything

No exceptions.

---

## 8. Mental model to keep

Training is historical.
Inference is contractual.

You do not guess what a model expects.
You enforce it.

---

## 9. Final note to future self

This project failed temporarily because feature contracts were implicit.

It now works because contracts are explicit.

Do not undo that.

If something feels wrong:
- inspect feature lists
- inspect shapes
- inspect contracts

Not vibes. Not assumptions.


