import torch.nn as nn

def build_criterion(config):
    loss_config = config["loss"]

    if loss_config["name"].lower() == "cross_entropy":
        return nn.CrossEntropyLoss()

    raise ValueError(
        f"Unsupported loss: {loss_config['name']}"
    )