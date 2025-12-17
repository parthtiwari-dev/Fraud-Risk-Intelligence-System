import pandas as pd
from src.models import predict

df = pd.read_csv("data/raw/creditcard.csv")

sample = df.iloc[100].to_dict()  
# or keep Class if your pipeline expects it

print(predict(sample))
