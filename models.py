import torch
import torch.nn as nn


# ==========================================
# 1. MLP MODEL
# Architecture: 784 -> 256 -> 128 -> 10
# ==========================================

class MLP(nn.Module):
    def __init__(self):
        super().__init__()

        self.network = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        return self.network(x)


# ==========================================
# 2. CNN MODEL
# Architecture:
# Conv2d(1,16,3,padding=1) -> ReLU -> MaxPool2d
# Conv2d(16,32,3,padding=1) -> ReLU -> MaxPool2d
# Flatten -> Linear(32*7*7,10)
# ==========================================

class CNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.network = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Flatten(),
            nn.Linear(32 * 7 * 7, 10)
        )

    def forward(self, x):
        return self.network(x)


# ==========================================
# 3. ATTENTION MODEL
# 28x28 image -> 16 patches of 7x7
# Embedding dimension: 64
# Transformer encoder layers: 2
# ==========================================

class AttentionModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.patch_embedding = nn.Conv2d(
            in_channels=1,
            out_channels=64,
            kernel_size=7,
            stride=7
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=64,
            nhead=4,
            dim_feedforward=128,
            batch_first=True
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=2
        )

        self.classifier = nn.Linear(64, 10)

    def forward(self, x):
        # Convert the image into 16 patch embeddings
        x = self.patch_embedding(x)

        # Convert feature maps into a sequence
        x = x.flatten(2)
        x = x.transpose(1, 2)

        # Process patches using Transformer encoder layers
        x = self.transformer(x)

        # Aggregate patch representations
        x = x.mean(dim=1)

        # Predict one of 10 classes
        x = self.classifier(x)

        return x