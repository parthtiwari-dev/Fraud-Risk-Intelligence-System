# FRIS - Fraud Risk Intelligence System

FRIS is an end to end fraud risk scoring system built on top of public credit card transaction data. It ingests transaction like JSON, applies feature engineering and a stack of supervised and unsupervised models, and returns a fraud risk score between 0 and 1 together with the top feature attributions that drove the decision.

## Goals

FRIS v1 will do the following:

- Load and explore real world credit card fraud data
- Engineer fraud specific features, including time patterns, frequency, rolling behaviour and anomaly scores
- Train supervised models (Logistic Regression, RandomForest, XGBoost) and compare them using fraud appropriate metrics
- Train unsupervised and neural components (IsolationForest, clustering, MLP or autoencoder) and combine them into an ensemble
- Produce explanations for each prediction using SHAP, with clear failure case analysis
- Serve the final model behind a FastAPI microservice that accepts transaction JSON and returns a risk score and top features
- Package the service in Docker and build a Streamlit UI that calls the API and visualises risk, SHAP explanations and a PCA cluster view

## Dataset

Primary dataset: Kaggle credit card fraud detection (European card transactions).

- Highly imbalanced target, fraud is a tiny fraction of all rows
- Columns: Time, V1 to V28, Amount, Class

The raw CSV is stored under `data/raw/creditcard.csv`.

## High level architecture

At a high level FRIS has the following pieces:

1. Data and features
   - Load raw CSV into a dataframe
   - Apply feature engineering and save processed train and test sets

2. Modelling
   - Train baseline models
   - Train boosted models and unsupervised blocks
   - Build a stacked ensemble

3. Explainability
   - Compute global SHAP importances
   - For each prediction, compute top k features that pushed it towards fraud or non fraud

4. Serving
   - Package preprocessing and model into an inference pipeline
   - Expose a FastAPI endpoint that takes JSON input and returns a risk score plus feature attributions

5. Deployment and UI
   - Use Docker to containerize the API
   - Build a Streamlit app that hits the API and shows an interactive fraud dashboard

## Project structure

See the `fris/` folder tree for modules, notebooks, API code and experiments.
