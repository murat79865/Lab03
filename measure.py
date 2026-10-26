import torch
import time
import numpy as np

from models import MLP, CNN, AttentionModel


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

models = {
    "MLP": MLP(),
    "CNN": CNN(),
    "Attention": AttentionModel()
}


def measure_latency(model, batch_size, runs):
    model.eval()

    x = torch.randn(batch_size, 1, 28, 28).to(device)
    model = model.to(device)

    times = []

    with torch.no_grad():
        for _ in range(10):
            model(x)

        for _ in range(runs):
            start = time.perf_counter()
            model(x)

            if device.type == "cuda":
                torch.cuda.synchronize()

            end = time.perf_counter()
            times.append((end - start) * 1000)

    return np.median(times)


for name, model in models.items():

    model = model.to(device)

    print("\n====================")
    print("MODEL:", name)
    print("====================")

    params = sum(p.numel() for p in model.parameters())

    print("Parameters:", params)

    latency_batch1 = measure_latency(model, 1, 100)
    latency_batch128 = measure_latency(model, 128, 20)

    print("Batch 1 latency:", round(latency_batch1, 4), "ms")
    print("Batch 128 latency:", round(latency_batch128, 4), "ms")