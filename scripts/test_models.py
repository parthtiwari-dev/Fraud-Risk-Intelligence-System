from src.models import predict

sample = {
    "Time": 100000,
    "Amount": 42.5
}

print(predict(sample))
