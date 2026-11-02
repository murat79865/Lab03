import matplotlib.pyplot as plt


# ==========================================
# MODEL RESULTS
# ==========================================

models = ["MLP", "CNN", "Attention"]

parameters = [
    235146,
    20490,
    70794
]

accuracy = [
    87.52,
    89.65,
    83.83
]


# ==========================================
# GRAPH 1: ACCURACY VS PARAMETERS
# ==========================================

plt.figure(figsize=(8, 5))

plt.scatter(parameters, accuracy)

for i, model in enumerate(models):
    plt.annotate(
        model,
        (parameters[i], accuracy[i])
    )

plt.xlabel("Number of Parameters")
plt.ylabel("Test Accuracy (%)")

plt.title("Test Accuracy vs Model Parameters")

plt.grid(True)

plt.savefig("accuracy_vs_parameters.png")

plt.show()


# ==========================================
# GRAPH 2: MODEL ACCURACY
# ==========================================

plt.figure(figsize=(8, 5))

plt.bar(models, accuracy)

plt.xlabel("Model")
plt.ylabel("Test Accuracy (%)")

plt.title("Fashion-MNIST Test Accuracy")

plt.savefig("model_accuracy.png")

plt.show()


print("Graphs saved successfully!")
# ==========================================
# GRAPH 3: INFERENCE LATENCY
# ==========================================

latency_batch1 = [
    0.0713,
    0.1676,
    0.6220
]

latency_batch128 = [
    0.2511,
    7.2368,
    2.8753
]

plt.figure(figsize=(8, 5))

plt.bar(
    models,
    latency_batch1
)

plt.xlabel("Model")
plt.ylabel("Latency (ms)")

plt.title("Inference Latency - Batch Size 1")

plt.savefig("latency_batch1.png")

plt.show()


plt.figure(figsize=(8, 5))

plt.bar(
    models,
    latency_batch128
)

plt.xlabel("Model")
plt.ylabel("Latency (ms)")

plt.title("Inference Latency - Batch Size 128")

plt.savefig("latency_batch128.png")

plt.show()


print("Latency graphs saved successfully!")
# ==========================================
# GRAPH 4 & 5: ACCURACY VS LATENCY
# ==========================================

for lat, bs in [(latency_batch1, 1), (latency_batch128, 128)]:
    plt.figure(figsize=(8, 5))
    plt.scatter(lat, accuracy)

    for i, model in enumerate(models):
        plt.annotate(model, (lat[i], accuracy[i]))

    plt.xlabel(f"Latency (ms), Batch Size {bs}")
    plt.ylabel("Test Accuracy (%)")
    plt.title(f"Test Accuracy vs Inference Latency (Batch {bs})")
    plt.grid(True)
    plt.savefig(f"accuracy_vs_latency_batch{bs}.png")
    plt.show()

print("Accuracy vs latency graphs saved successfully!")