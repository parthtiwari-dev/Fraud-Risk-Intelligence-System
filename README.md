# FRIS - Fraud Risk Intelligence System

**One-line pitch for resume**
End-to-end Fraud Risk Intelligence System: data ingestion, feature engineering, supervised and unsupervised models, ensemble risk scoring, SHAP explainability, FastAPI serving, Dockerized deployment, and Streamlit demo.

## Project goal

Build a production-style end-to-end ML system that:

1. Is deployable and reproducible.
2. Can be explained for 30-40 minutes in an interview.
3. Gets shortlisted for ₹12 LPA+ ML/AI engineer roles.

Secondary outputs:

* Complete GitHub repo with notebooks and scripts.
* Dockerized FastAPI inference service.
* Streamlit demo UI.
* Screencast demo and interview cheat sheet.

## Repo layout

```
fris/
├─ data/
│  ├─ raw/
│  ├─ processed/
├─ notebooks/
│  ├─ 01_EDA.ipynb
│  ├─ 02_feature_engineering.ipynb
│  ├─ 03_models_baselines.ipynb
│  ├─ 04_nn_autoencoder.ipynb
├─ src/
│  ├─ data_loader.py
│  ├─ features.py
│  ├─ models.py
│  ├─ train.py
│  ├─ infer.py
│  ├─ explain.py
│  ├─ api/
│  │  ├─ main.py
│  │  ├─ schemas.py
├─ app/
│  ├─ streamlit_app.py
├─ experiments/
│  ├─ models/
│  ├─ figures/
├─ services/
│  ├─ docker/
│  │  ├─ Dockerfile
├─ README.md
├─ requirements.txt
```

## Quickstart - development

**Run these commands from repo root**.

### Clone

```
git clone <your-repo-url> fris
cd fris
```

### Environment setup

```
python3.10 -m venv .venv
. .venv/Scripts/Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
```

### Jupyter for notebooks

```
jupyter lab
```

### Training scripts

```
python src/train.py --stage all
```

### Run FastAPI locally

```
uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000
```

### Run Streamlit UI

```
streamlit run app/streamlit_app.py
```

## Quickstart - Docker

```
docker build -t fris:latest -f services/docker/Dockerfile .
docker run --rm -p 8000:8000 -p 8501:8501 fris:latest
```

API on `http://localhost:8000`
Streamlit on `http://localhost:8501`

## Example curl

```
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "transaction_id": "tx_0001",
    "timestamp": "2024-09-01T14:23:05Z",
    "amount": 123.45,
    "merchant_id": "m_1245",
    "device_id": "d_998",
    "device_type": "mobile",
    "geo_bucket": "city_12",
    "account_id": "a_55",
    "account_age_days": 45,
    "card_present": false,
    "currency": "INR"
  }'
```

### Example response

```
{
  "transaction_id": "tx_0001",
  "score": 0.8723,
  "label": "fraud",
  "top_features": [
    {"name": "amount_log", "value": 4.828, "shap": 0.42},
    {"name": "merchant_freq", "value": 0.02, "shap": 0.12},
    {"name": "is_new_merchant", "value": 1, "shap": 0.08}
  ],
  "model_version": "xgboost_v1.joblib",
  "debug": {
    "anomaly_score": -0.12,
    "pca_cluster": 2
  }
}
```

## Data and augmentation

* Base dataset: Kaggle Credit Card Fraud Detection.
* Synthetic additions: `merchant_id`, `device_type`, `geo_bucket`, `account_age_days`, account history.
* Processed CSV: `data/processed/train.csv`, `data/processed/test.csv`.
* Saved transformers in `experiments/models/`.

## Feature engineering (minimum set)

* Time: hour, dayofweek
* Amount: log(amount+1), scaled amount
* Frequencies: merchant_freq, device_freq, account_txn_count
* Rolling: last_5_mean_amount, last_5_count
* Encodings: frequency or target encoding
* Missing flags
* Interactions: amount * account_age_days, is_new_merchant
* PCA (2 components)
* IsolationForest anomaly score

## Models

* LogisticRegression
* DecisionTree
* RandomForest
* XGBoost
* MLPClassifier
* IsolationForest
* Stacking meta learner

Metrics saved: Precision, Recall, F1 (fraud), AUC-ROC, AUC-PR.

## Explainability

* SHAP TreeExplainer.
* Local top features in API response.
* Global feature importance in `experiments/figures/`.

## Streamlit demo

* Input transaction JSON
* Calls API
* Shows SHAP, PCA position, similar transactions

## Tests

```
python src/api/test_client.py
```

## Timeline (10 days)

**Day 0** repo skeleton, requirements, dataset sample, stub files.
**Day 1** EDA notebook + figures.
**Day 2** feature pipeline + processed data + joblib.
**Day 3** baseline models.
**Day 4** XGBoost + CV.
**Day 5** IsolationForest + PCA plots.
**Day 6** MLP + stacking.
**Day 7** SHAP global + local explanations.
**Day 8** FastAPI + inference pipeline.
**Day 9** Dockerfile + container tests.
**Day 10** Streamlit UI.

## Requirements

```
pandas>=1.5
numpy>=1.24
scikit-learn>=1.2
xgboost>=1.7
shap>=0.41
fastapi>=0.95
uvicorn[standard]>=0.22
joblib>=1.2
streamlit>=1.20
matplotlib>=3.7
seaborn>=0.12
python-dateutil
tqdm
```

## Reproducibility

* Fixed seeds
* Saved split indices
* Saved preprocessor
* Saved model versions

## License

MIT.
