# FEATURES PIPELINE CONTRACT

- **File:** `docs/FEATURES_CONTRACT.md`
- **Source of truth:** `src/features.py`
- **Status:** READ-ONLY for Day 8–9

---

## 0. HARD RULES (NON-NEGOTIABLE)

1. **`features.py` is the single source of truth.**
2. No logic duplication in notebooks or API.
3. No `validate_input()` anywhere.
4. No shortcuts for inference.
5. **Training and inference must use the same pipeline.**
6. Everything starts from raw input.
7. **One path only:** `raw → feature_pipeline → model`

> **Note:** If any future code violates this, it is wrong.

---

## 1. PURPOSE OF `features.py`

`features.py` defines the entire feature engineering contract used during training. It answers exactly three questions:

1. What raw columns are required?
2. What engineered columns are produced?
3. What fitted objects are needed at inference?

Nothing else matters.

---

## 2. ENTRY POINT: `feature_pipeline`

This is the only function that matters. Everything else is an internal step.

df_final, preprocessors = feature_pipeline(
df,
fit=True | False,
preprocessors=...
)


---

## 3. REQUIRED RAW INPUT COLUMNS (ABSOLUTE MINIMUM)

From reading the code, the pipeline assumes these raw columns exist before step 1. If they don’t, the pipeline crashes immediately.

| Column | Type | Description |
| :--- | :--- | :--- |
| `Time` | `int/float` | Seconds since reference |
| `Amount` | `float` | Transaction amount |

---

## 4. SYNTHETICALLY GENERATED COLUMNS (NOT API INPUT)

These columns are generated **inside** the pipeline, NOT provided by API:

- `merchant_id`
- `device_type`
- `geo_bucket`
- `account_id`
- `account_age_days`

**Important Implication:**
- Your API does **NOT** need to send these today.
- They are generated deterministically using a seed.
- This is acceptable for a simulated dataset (in real production, these would come from upstream systems).
- **Do NOT try to “validate” these at the API level.**

---

## 5. STEP-BY-STEP PIPELINE (ORDER IS CRITICAL)

This is the exact execution order inside `feature_pipeline`.

### Step 1 — Timestamp features
*   **Input:** `Time`
*   **Output:** `timestamp`, `hour`, `dayofweek`

### Step 2 — Amount features
*   **Input:** `Amount`
*   **Output:** `amount_log`, `amount_scaled`
*   **Uses:** `RobustScaler` (Saved as `preprocessors["scaler"]`)

### Step 3 — Synthetic categories
*   **Output (generated):** `merchant_id`, `device_type`, `geo_bucket`, `account_id`, `account_age_days`
*   **Mechanism:** Seeded RNG → deterministic.

### Step 4 — Frequency features
*   **Source:** Derived from synthetic categories
*   **Output:** `merchant_freq`, `device_freq`, `account_txn_count`

### Step 5 — Rolling behavioral features
*   **Source:** Computed per `account_id`
*   **Requires:** Sorting by `account_id` and `timestamp`
*   **Output:** `last_5_mean_amount`, `last_5_count`

### Step 6 — Missing flags
*   **Output (Boolean indicators):**
    *   `merchant_id_missing`
    *   `device_type_missing`
    *   `geo_bucket_missing`
    *   `account_age_days_missing`

### Step 7 — Categorical encodings
*   **Method:** Frequency encoding
*   **Output:** `merchant_id_fe`, `device_type_fe`, `geo_bucket_fe`, `account_id_fe`
*   **Uses:** `preprocessors["encoders"]`

### Step 8 — Interaction features
*   **Output:** `amount_times_age`, `is_new_merchant`

### Step 9 — PCA
*   **Scope:** Applies PCA on all numeric columns at that moment.
*   **Output:** `pca_x`, `pca_y`
*   **Uses:** `preprocessors["pca"]`

---

## 6. FINAL OUTPUT COLUMNS (IMPORTANT)

`feature_pipeline` returns a DataFrame containing:
1. Original raw columns
2. All engineered columns above
3. PCA columns

This output DF is the **training-time feature universe**. Anything not in this DF must not be referenced later.

---

## 7. PREPROCESSORS OBJECT

Returned as a dictionary and must be reused during inference.

preprocessors = {
"scaler": RobustScaler,
"encoders": dict,
"pca": PCA
}

**Persistence:**
Saved to disk using `save_preprocessors(preprocessors)`.
