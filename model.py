import torch.nn as nn

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
