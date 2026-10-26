
import torch
from torchinfo import summary
from thop import profile

from models import MLP, CNN, AttentionModel

models = {
    "MLP": MLP(),
    "CNN": CNN(),
    "Attention": AttentionModel()
}

for name, model in models.items():

    print("\n====================")
    print("MODEL:", name)
    print("====================")

    # Parameter summary
    summary(
        model,
        input_size=(1, 1, 28, 28)
    )

    # FLOPs calculation
    input_tensor = torch.randn(1, 1, 28, 28)

    macs, params = profile(
        model,
        inputs=(input_tensor,),
        verbose=False
    )

    print("Parameters:", params)
    print("MACs:", macs)
    print("Estimated FLOPs:", macs * 2)
