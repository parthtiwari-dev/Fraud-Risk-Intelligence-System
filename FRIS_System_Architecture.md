# FRIS System Architecture: Complete Technical Contract

**Status:** Source of Truth for Days 8–14  
**Last Updated:** December 17, 2025  
**Scope:** Training pipeline → Inference layer → Explainability  

---

## Part 1: Global System View

### The Three Worlds (Current State)

FRIS exists as three disconnected systems:

1. **Training World** (`features.py`)
   - Transforms raw data into engineered features
   - Not used anywhere else in the current codebase
   - Source of truth for feature definitions

2. **Inference World** (`models.py`)
   - Loads trained models and makes predictions
   - Assumes engineered features already exist
   - No connection to `features.py`

3. **Explainability World** (`explain.py`)
   - Generates SHAP explanations for predictions
   - Depends on engineered features
   - Currently piggybacks on incorrect abstraction

**Critical Truth:** These three worlds never meet. The missing piece is `prepare_features()`, which will serve as the unified entry point for all inference paths.

### Non-Negotiable Rules

1. `features.py` is the single source of truth for feature engineering
2. No logic duplication in notebooks, API, or other modules
3. Training and inference must use identical pipelines
4. Everything starts from raw input
5. One path only: `raw input → feature_pipeline → models → output`
6. No shortcuts, no workarounds, no exceptions

---

## Part 2: Feature Engineering Contract (`src/features.py`)

### Purpose

`features.py` is the **only legitimate place** where raw transactional data is converted into engineered features. Training-time truth lives here.

**It answers exactly three questions:**
1. What raw columns are required?
2. What engineered columns are produced?
3. What fitted objects are needed at inference?

### Core Entry Point

```python
df_final, preprocessors = feature_pipeline(
    df,
    fit=True | False,
    preprocessors=None
)
```

### Input Contract

The function expects a pandas DataFrame containing at minimum:

| Column | Type | Description |
|--------|------|-------------|
| `Time` | int/float | Seconds since reference epoch |
| `Amount` | float | Transaction amount in base currency |

**Critical:** Missing columns will cause silent failures or crashes. There are no guardrails.

### Synthetic Column Generation

The following columns are **generated deterministically inside the pipeline** using a seeded RNG:

- `merchant_id`
- `device_type`
- `geo_bucket`
- `account_id`
- `account_age_days`

**Why this matters:** Your API does **NOT** need to send these. They are synthetic in the current simulated system. In production, they would come from upstream systems (merchant database, device profiles, geo service, customer database).

### Pipeline Execution Order (Deterministic)

**Step 1: Timestamp Features**
- Input: `Time`
- Output: `timestamp` (formatted), `hour` (0–23), `dayofweek` (0–6)

**Step 2: Amount Features**
- Input: `Amount`
- Output: `amount_log`, `amount_scaled`
- Uses: `RobustScaler` (saved to `preprocessors["scaler"]`)
- Reason: Log transformation handles skew; robust scaling resists outliers

**Step 3: Synthetic Category Generation**
- Method: Seeded RNG ensures deterministic output
- Output: `merchant_id`, `device_type`, `geo_bucket`, `account_id`, `account_age_days`
- Implication: Same seed = same values for all training runs

**Step 4: Frequency Features**
- Source: Counts derived from synthetic categories in historical dataset
- Output: `merchant_freq`, `device_freq`, `account_txn_count`
- Mechanism: For each row, count historical occurrences of its category

**Step 5: Rolling Behavioral Features**
- Prerequisite: Data must be sorted by `account_id` then `timestamp`
- Output: `last_5_mean_amount`, `last_5_count`
- Computation: Per-account rolling window (5 preceding transactions)
- Implication: First 5 transactions per account have partial windows

**Step 6: Missing Indicators**
- Output: Boolean flags for potential missing data
  - `merchant_id_missing`
  - `device_type_missing`
  - `geo_bucket_missing`
  - `account_age_days_missing`

**Step 7: Categorical Encodings**
- Method: Frequency encoding (category count / total observations)
- Output: `merchant_id_fe`, `device_type_fe`, `geo_bucket_fe`, `account_id_fe`
- Uses: `preprocessors["encoders"]` (dict mapping category → frequency)
- Why: Target encoding without data leakage; handles unseen categories gracefully

**Step 8: Interaction Features**
- Output: 
  - `amount_times_age` (Amount × account_age_days)
  - `is_new_merchant` (boolean: merchant_freq == 1)

**Step 9: PCA (Dimensionality Reduction)**
- Scope: Applied to all numeric columns at this point
- Output: `pca_x`, `pca_y` (first two principal components)
- Uses: `preprocessors["pca"]` (fitted PCA object)
- Purpose: Visualization and redundancy reduction

### Complete Output Schema

The returned DataFrame contains:

**Original Raw Columns:**
- `Time`, `Amount`

**Time-Derived:**
- `timestamp`, `hour`, `dayofweek`

**Amount-Derived:**
- `amount_log`, `amount_scaled`

**Synthetic Categories:**
- `merchant_id`, `device_type`, `geo_bucket`, `account_id`, `account_age_days`

**Frequency Features:**
- `merchant_freq`, `device_freq`, `account_txn_count`

**Behavioral Features:**
- `last_5_mean_amount`, `last_5_count`

**Missing Flags:**
- `merchant_id_missing`, `device_type_missing`, `geo_bucket_missing`, `account_age_days_missing`

**Encoded Categories:**
- `merchant_id_fe`, `device_type_fe`, `geo_bucket_fe`, `account_id_fe`

**Interactions:**
- `amount_times_age`, `is_new_merchant`

**Dimensionality Reduction:**
- `pca_x`, `pca_y`

### What `features.py` Does NOT Do

- Run ML models
- Compute probabilities or anomaly scores
- Generate cluster IDs
- Create SHAP values
- Perform any inference
- Make any decisions

**These belong in `models.py` and `explain.py`.**

### Preprocessors Object

Returned as a dictionary and reused during inference:

```python
preprocessors = {
    "scaler": RobustScaler(),         # Fitted on training data
    "encoders": {                     # Dict of dicts
        "merchant_id_fe": {...},      # {category: frequency}
        "device_type_fe": {...},
        "geo_bucket_fe": {...},
        "account_id_fe": {...}
    },
    "pca": PCA(n_components=2)        # Fitted on training data
}
```

**Persistence:** Must be saved to disk and reloaded at inference time. Do not refit on inference data.

### Training Contract (Critical)

Whatever columns exist in the output DataFrame during training must be identical at inference:
- Same column names
- Same column order (optional but recommended)
- Same data types
- Same preprocessing objects

**If inference does not call this pipeline, the system is invalid.**

---

## Part 3: Model Inference Contract (`src/models.py`)

### Purpose

`models.py` consumes engineered features and produces:
- Base model signals (probabilities, anomaly scores, reconstruction errors)
- Meta-features (inputs to stacking layer)
- Final fraud score and label

**Critical:** It must NEVER perform feature engineering.

### Load Models Function

```python
models_dict = load_models()
```

Returns a dictionary containing:

**Frozen Models:**
- `xgb` — Trained XGBoost classifier
- `iforest` — Trained IsolationForest
- `autoencoder` — Trained deep neural network
- `stacker` — Trained meta-learner

**Feature Contracts:**
- `xgb_features` — List of columns XGBoost expects
- `iforest_features` — List of columns IsolationForest expects
- `ae_features` — List of columns Autoencoder expects
- `meta_features` — List of features stacker expects (in exact order)

**Metadata:**
- `threshold` — Float for converting probability to label
- `required_input_features` — Complete list of engineered features required

**Critical Assumption:** This function assumes training artifacts already exist and are correctly serialized. It performs no validation.

### Base Model Signals (Generated at Inference Time)

These columns are **NOT** engineered features. They are model outputs computed during inference:

**XGBoost Signal:**
- `xgb_proba` — Fraud probability from XGBoost
- Computation: `xgb.predict_proba(engineered_df[xgb_features])[:, 1]`
- Range: [0, 1]

**IsolationForest Signal:**
- `anomaly_score` — Inverted decision function
- Computation: `-iforest.decision_function(engineered_df[iforest_features])`
- Range: Typically [-1, 1]
- Why inverted: Higher score = more anomalous (fraud-like)

**Autoencoder Signal:**
- `ae_recon_error` — Mean squared reconstruction error
- Computation: `MSE(original_input, autoencoder(original_input))`
- Range: [0, ∞)
- Why: High reconstruction error suggests anomalous pattern

**Autoencoder Latent Features (Conceptual):**
- `ae_latent_1` through `ae_latent_8` (if used)
- Computation: `z = autoencoder.encoder(X)`
- Purpose: Learned representations passed to stacker
- Note: Only included if explicitly used during training

### Cluster ID (Important Clarification)

**Cluster ID is:**
- NOT raw data
- NOT a feature engineered by `features.py`
- OUTPUT of a trained clustering model applied to engineered features

**Conceptual pipeline:**
```
engineered_df → clustering_model.predict() → cluster_id
```

**Responsibility:** Belongs in `models.py`, not `features.py`

### Input Validation (Misnamed Function)

```python
validate_input(input_dict) → single_row_df
```

**Current Reality:**
- Expects a FULLY ENGINEERED feature dictionary
- Checks presence of every column required by models
- Ensures all values are numeric
- Returns a single-row DataFrame

**What it does NOT do:**
- Perform feature engineering
- Accept raw transaction input
- Call `features.py`
- Transform any columns

**Why it's misnamed:** This is not validation of raw input. This is validation of post-feature-engineering output.

**Critical:** This function will be removed in the inference layer redesign. Replace it with `prepare_features()`.

### Base Signals Computation

```python
compute_base_signals(input_df) → dict
```

**Input:** Single-row DataFrame, already engineered, column-perfect

**Pipeline:**
1. Extract XGBoost features
2. Compute XGBoost probability
3. Extract IsolationForest features
4. Compute anomaly score
5. Extract Autoencoder features
6. Compute reconstruction error

**Output:** Dictionary containing all three signals

**Example:**
```python
{
    "xgb_proba": 0.87,
    "anomaly_score": 0.42,
    "ae_recon_error": 0.15
}
```

### Meta-Features Assembly

```python
build_meta_features(input_dict) → numpy_array
```

**Current Pipeline:**
1. Calls `validate_input(input_dict)` ← **Will be removed**
2. Computes base signals
3. Manually assembles meta-feature row
4. Orders features exactly as stacker expects
5. Returns NumPy array

**Critical Implication:** `build_meta_features` cannot work on raw input. It assumes engineered features already exist.

**Meta-Feature Composition:**

| Feature | Source | Type |
|---------|--------|------|
| `xgb_oof_proba` | Base signal | float |
| `anomaly_score` | Base signal | float |
| `ae_recon_error` | Base signal | float |
| `cluster_id` | Inference | int |
| `amount_log` | Engineered | float |
| `merchant_freq` | Engineered | float |
| `account_txn_count` | Engineered | int |
| `last_5_mean_amount` | Engineered | float |

**Order matters:** Stacker was trained on features in a specific sequence. Different order = wrong predictions.

### Final Prediction

```python
predict_proba(input_dict) → float
```

**Pipeline:**
1. Build meta-features
2. Pass to stacker
3. Return fraud probability
4. Range: [0, 1]

```python
predict(input_dict) → dict
```

**Pipeline:**
1. Get probability via `predict_proba()`
2. Apply threshold (default: 0.5 or from config)
3. Assign label

**Output:**
```python
{
    "score": 0.87,
    "label": "fraud"  # "fraud" | "legit"
}
```

**Critical:** This is the only place labels are assigned.

### Current State Summary

- Logically consistent internally
- Correctly implements base model inference
- Correctly stacks predictions
- Externally unusable for raw input
- No connection to `features.py`
- This is **not a bug**—it's an unfinished inference layer

---

## Part 4: Explainability Contract (`src/explain.py`)

### Purpose

Explain why XGBoost produced its score for a transaction using SHAP (SHapley Additive exPlanations).

### Current State (Facts)

- SHAP math is correct
- TreeExplainer is properly instantiated
- Feature attribution logic is sound
- Input contract is wrong
- Integration violates system architecture

### Load Explainer

```python
load_explainer(xgb_model) → TreeExplainer
```

**Pipeline:**
1. Takes trained XGBoost model
2. Creates `shap.TreeExplainer`
3. Returns explainer object

**Usage:** Call once at startup, cache in memory. No issues here.

### Compute SHAP for Single Row

```python
compute_shap_single(explainer, xgb_df) → shap_values
```

**Input:** Single-row DataFrame with exact XGBoost features

**Pipeline:**
1. Explainer processes row
2. Returns SHAP values for each feature
3. One value per feature

**Properties:**
- Sums to difference between prediction and base value
- Positive = pushes toward fraud
- Negative = pushes toward legit
- This function is correct

### Extract Top-K Features

```python
top_k_features(shap_values, xgb_df, k=5) → list
```

**Output:**
```python
[
    {"feature": "amount_log", "shap_value": 0.42, "actual_value": 2.15},
    {"feature": "merchant_freq", "shap_value": 0.18, "actual_value": 5},
    ...
]
```

**Properties:**
- JSON-safe
- Ordered by absolute SHAP value
- Includes actual feature values for context
- API-ready
- This function is solid

### Current Explain Transaction (Broken)

```python
explain_transaction(input_dict, models, explainer) → dict
```

**Current Pipeline:**
1. Calls `validate_input(input_dict)` ← **Wrong**
2. Slices XGBoost features
3. Computes SHAP
4. Returns top-k

**Hidden Assumptions:**
- `input_dict` already contains engineered features
- Raw input is impossible

**Why it's wrong:**
- Violates raw input principle
- Violates one-path rule
- Depends on incorrect function

### Correct Conceptual Flow (Post-Refactor)

```
raw_dict
  ↓
prepare_features() ← features.py
  ↓
engineered_df
  ↓
engineered_df[xgb_features]
  ↓
compute_shap_single()
  ↓
SHAP values
```

### What Explain MUST NOT Do

- No feature engineering
- No input validation
- No model inference beyond XGB
- No thresholding
- No decision logic

**Scope:** Explain exactly what the model saw, nothing more.

---

## Part 5: Integration Rules (The Missing Piece)

### Where `prepare_features()` Will Live

**Location:** `src/inference.py` (new file, Day 9+)

**Responsibility:** Single entry point for all inference paths

**Purpose:** Transforms raw input through `features.py` pipeline

### Unified Pipeline (Correct Flow)

```
raw_dict (API input)
  ↓
prepare_features()
  ├─ Load preprocessors
  ├─ Convert dict to DataFrame
  ├─ Call feature_pipeline()
  └─ Return engineered_df
  ↓
models.py
  ├─ predict()
  ├─ predict_proba()
  └─ build_meta_features()
  ↓
explain.py
  ├─ compute_shap_single()
  └─ top_k_features()
  ↓
API Response
```

### One Path Rule

Every inference request follows: `raw → prepare_features → model → output`

**Implications:**
- No shortcuts
- No duplicate logic
- No direct calls to `validate_input()` in serving
- All feature engineering centralized

---

## Part 6: Daily Checklist (Read Every Morning)

✅ **`features.py`** = Definition of what engineered features look like  
✅ **`models.py`** = What trained models think about engineered features  
✅ **Stacker** = Final decision based on base model signals  
✅ **`explain.py`** = Why the decision happened using SHAP  

❌ **If any file crosses its boundary, stop and fix it.**

---

## Part 7: Current Gaps (Honest Assessment)

| Component | Status | Issue |
|-----------|--------|-------|
| Feature engineering pipeline | ✅ Complete | Not used in inference |
| Model inference | ✅ Complete | Assumes engineered input magically exists |
| SHAP explanation | ✅ Complete | Tightly coupled to wrong abstraction |
| API layer | ❌ Missing | No `prepare_features()` |
| End-to-end pipeline | ❌ Missing | No connection between layers |
| Raw-to-prediction | ❌ Missing | Raw input path not implemented |

**Next milestone:** Implement `prepare_features()` to unify all layers.

---

## Appendix: Feature Order Reference

This is the exact column order expected by trained models. Do not reorder.

### XGBoost Feature Order
See `load_models()["xgb_features"]` at runtime.

### IsolationForest Feature Order
See `load_models()["iforest_features"]` at runtime.

### Autoencoder Feature Order
See `load_models()["ae_features"]` at runtime.

### Stacking Meta-Features Order
See `load_models()["meta_features"]` at runtime.

**Why order matters:** Trained models rely on positional features. Wrong order = completely wrong predictions.

---

## Appendix: Preprocessors Persistence

### Save During Training
```python
save_preprocessors(preprocessors, path="models/preprocessors.pkl")
```

### Load During Inference
```python
preprocessors = load_preprocessors(path="models/preprocessors.pkl")
```

### Never Refit at Inference
Preprocessors are fit on training data only. Do not apply `fit()` on new data.

---

**End of Document**  
**For questions about boundaries or contracts, reference Part 1 (Non-Negotiable Rules).**