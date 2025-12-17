from src.models import load_models
from src.explain import load_explainer, explain_transaction
import pandas as pd
from pathlib import Path

DATA_DIR = Path("data/processed")
test_df = pd.read_csv(DATA_DIR / "test.csv")

models = load_models()
explainer = load_explainer(models["xgb"])

sample = test_df.drop(columns=["Class"]).iloc[0].to_dict()

explanation = explain_transaction(sample, models, explainer, k=5)
print(explanation)
