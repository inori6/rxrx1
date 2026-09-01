import torch

def build_optimizer(model, config):
    optimizer_config = config["optimizer"]

    if optimizer_config["name"].lower() == "adamw":
        return torch.optim.AdamW(
            model.parameters(),
            lr=optimizer_config["lr"],
            weight_decay=optimizer_config["weight_decay"],
        )

    raise ValueError(
        f"Unsupported optimizer: {optimizer_config['name']}"
    )