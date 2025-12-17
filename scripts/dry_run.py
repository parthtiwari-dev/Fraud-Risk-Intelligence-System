import pandas as pd
from src.models import load_models, predict
from src.explain import load_explainer, explain_transaction

df = pd.read_csv("data/raw/creditcard.csv")


models = load_models()
explainer = load_explainer(models["xgb"])


for idx in [1, 10, 100, 500, 1000, 20000, 40000, 60000, 80000, 100000, 150000, 175000, 200000, 220212, 250000]:
    raw = df.iloc[idx].to_dict()
    print(idx, predict(raw))
    print(explain_transaction(raw, models, explainer))
