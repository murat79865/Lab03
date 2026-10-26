import torch
import torch.nn as nn


# ==========================================
# 1. MLP MODEL
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
# ==========================================

class AttentionModel(nn.Module):
    def __init__(self):
        super().__init__()

        # Divide image into 7x7 patches
        self.patch_embedding = nn.Conv2d(
            1,
            64,
            kernel_size=7,
            stride=7
        )

        # Transformer encoder layer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=64,
            nhead=4,
            dim_feedforward=128,
            batch_first=True
        )

        # Two Transformer encoder layers
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=2
        )

        # Classification layer
        self.classifier = nn.Linear(64, 10)

    def forward(self, x):
        # Create patches
        x = self.patch_embedding(x)

        # Prepare sequence
        x = x.flatten(2)
        x = x.transpose(1, 2)

        # Transformer
        x = self.transformer(x)

        # Mean pooling
        x = x.mean(dim=1)

        # Classification
        x = self.classifier(x)

        return x