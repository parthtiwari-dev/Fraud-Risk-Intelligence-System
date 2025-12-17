import pandas as pd
import requests

df = pd.read_csv("data/raw/creditcard.csv")

# Pick a real transaction
row = df.iloc[100].to_dict()

resp = requests.post(
    "http://127.0.0.1:8000/predict",
    json=row
)

print("INPUT CLASS:", row.get("Class", "unknown"))
print("MODEL OUTPUT:", resp.json())


fraud_rows = df[df["Class"] == 1].sample(5)
legit_rows = df[df["Class"] == 0].sample(5)

for i, row in fraud_rows.iterrows():
    out = requests.post("http://127.0.0.1:8000/predict", json=row.to_dict()).json()
    print("TRUE: FRAUD | PRED:", out)

for i, row in legit_rows.iterrows():
    out = requests.post("http://127.0.0.1:8000/predict", json=row.to_dict()).json()
    print("TRUE: LEGIT | PRED:", out)
