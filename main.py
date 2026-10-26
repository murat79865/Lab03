
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim

from torchvision import transforms
from torch.utils.data import random_split, DataLoader

from models import MLP, CNN, AttentionModel


# ==========================================
# STEP 1: DEVICE
# ==========================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", device)


# ==========================================
# STEP 2: PREPARE DATASET
# ==========================================

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.2860,), (0.3530,))
])


train_dataset = torchvision.datasets.FashionMNIST(
    root="./data",
    train=True,
    download=True,
    transform=transform
)

test_dataset = torchvision.datasets.FashionMNIST(
    root="./data",
    train=False,
    download=True,
    transform=transform
)


# ==========================================
# STEP 3: SPLIT DATASET
# ==========================================

train_set, val_set = random_split(
    train_dataset,
    [55000, 5000],
    generator=torch.Generator().manual_seed(42)
)


# ==========================================
# STEP 4: CREATE DATALOADERS
# ==========================================

train_loader = DataLoader(
    train_set,
    batch_size=128,
    shuffle=True,
    num_workers=0
)

val_loader = DataLoader(
    val_set,
    batch_size=128,
    shuffle=False,
    num_workers=0
)

test_loader = DataLoader(
    test_dataset,
    batch_size=128,
    shuffle=False,
    num_workers=0
)


print("Training images:", len(train_set))
print("Validation images:", len(val_set))
print("Test images:", len(test_dataset))


# ==========================================
# STEP 5: DEFINE MODELS
# ==========================================

models = {
    "MLP": MLP(),
    "CNN": CNN(),
    "Attention": AttentionModel()
}


# ==========================================
# STEP 6: TRAINING FUNCTION
# ==========================================

def train_model(model, name):

    model = model.to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(
        model.parameters(),
        lr=0.001
    )

    epochs = 5

    for epoch in range(epochs):

        model.train()

        running_loss = 0.0

        for images, labels in train_loader:

            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(images)

            loss = criterion(outputs, labels)

            loss.backward()

            optimizer.step()

            running_loss += loss.item()

        average_loss = running_loss / len(train_loader)

        # Validation
        model.eval()

        correct = 0
        total = 0

        with torch.no_grad():

            for images, labels in val_loader:

                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)

                _, predicted = torch.max(outputs, 1)

                total += labels.size(0)

                correct += (
                    predicted == labels
                ).sum().item()

        val_accuracy = 100 * correct / total

        print(
            name,
            "Epoch:",
            epoch + 1,
            "/",
            epochs,
            "| Loss:",
            round(average_loss, 4),
            "| Validation Accuracy:",
            round(val_accuracy, 2),
            "%"
        )

    return model


# ==========================================
# STEP 7: TEST FUNCTION
# ==========================================

def test_model(model, name):

    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():

        for images, labels in test_loader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)

            correct += (
                predicted == labels
            ).sum().item()

    accuracy = 100 * correct / total

    print(
        name,
        "Test Accuracy:",
        round(accuracy, 2),
        "%"
    )

    return accuracy


# ==========================================
# STEP 8: TRAIN AND TEST ALL MODELS
# ==========================================

results = {}

for name, model in models.items():

    print("\n================================")
    print("TRAINING:", name)
    print("================================")

    trained_model = train_model(
        model,
        name
    )

    print("\nTesting:", name)

    accuracy = test_model(
        trained_model,
        name
    )

    results[name] = accuracy


# ==========================================
# STEP 9: SAVE RESULTS
# ==========================================

with open("results.txt", "w") as file:

    file.write("Fashion-MNIST Results\n")
    file.write("====================\n\n")

    for name, accuracy in results.items():

        file.write(
            f"{name}: {accuracy:.2f}%\n"
        )


print("\n================================")
print("ALL MODELS TRAINED AND TESTED")
print("================================")

print("Results saved to results.txt")
