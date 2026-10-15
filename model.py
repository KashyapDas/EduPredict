import torch
import torch.nn as nn
from pathlib import Path

# create the architecture of the model
def create_model(input_features):
    model = nn.Sequential(
        nn.Linear(input_features, 16),
        nn.ReLU(),
        nn.Linear(16, 8),
        nn.ReLU(),
        nn.Linear(8, 1)
    )

    return model


# Load the trained model
def load_model():
    model_path = Path(__file__).resolve().parent / "my_model.pt"

    checkpoint = torch.load(
        model_path,
        map_location="cpu",
        weights_only=True
    )

    input_features = checkpoint["input_features"]

    model = create_model(input_features)

    model.load_state_dict(checkpoint["model_state_dict"])

    model.eval()

    return model