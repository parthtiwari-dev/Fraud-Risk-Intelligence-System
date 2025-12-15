from src.models import predict, load_models

models = load_models()
features = models["required_input_features"]

sample = {f: 0.0 for f in features}
sample.update({
    "amount_log": 4.2,
    "merchant_freq": 12,
    "account_txn_count": 7,
    "last_5_mean_amount": 310.5,
    "cluster_id": 0
})

out = predict(sample)
print(out)
