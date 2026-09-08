import torch


_LR_RATIOS = {
    "1_1_3_10": (1.0, 1.0, 3.0, 10.0),
    "1_3_5_10": (1.0, 3.0, 5.0, 10.0),
    "1_3_10_30": (1.0, 3.0, 10.0, 30.0),
}


def _parse_lr_ratio(value):
    if isinstance(value, str):
        if value not in _LR_RATIOS:
            raise ValueError(f"Unsupported lr_ratio: {value}")

        return _LR_RATIOS[value]

    if isinstance(value, (list, tuple)):
        if len(value) != 4:
            raise ValueError("lr_ratio must contain exactly 4 values.")

        ratio = tuple(float(item) for item in value)

        if any(item <= 0 for item in ratio):
            raise ValueError("lr_ratio values must be greater than zero.")

        return ratio

    raise TypeError("lr_ratio must be a string, list, or tuple.")


def _build_discriminative_groups(
    model,
    base_lr,
    lr_ratio,
    fusion_lr_ratio=3.0,
    neck_lr_ratio=None,
):
    if not hasattr(model, "features"):
        raise ValueError("Discriminative learning rates require model.features.")
    if not hasattr(model, "classifier"):
        raise ValueError("Discriminative learning rates require model.classifier.")
    if len(model.features) < 9:
        raise ValueError("Expected model.features to contain at least 9 stages.")

    ratios = _parse_lr_ratio(lr_ratio)
    lrs = [base_lr * ratio for ratio in ratios]

    classifier_params = list(model.classifier.parameters())
    neck = getattr(model, "neck", None)
    neck_params = list(neck.parameters()) if neck is not None else []

    if neck_params and neck_lr_ratio is not None:
        neck_ids = {id(p) for p in neck_params}
        head_params = [p for p in classifier_params if id(p) not in neck_ids]
    else:
        head_params = classifier_params
        neck_params = []

    projection = getattr(model, "projection", None)
    if projection is not None:
        head_params += list(projection.parameters())

    groups = [
        {"params": model.features[0:3].parameters(), "lr": lrs[0], "group_name": "early"},
        {"params": model.features[3:6].parameters(), "lr": lrs[1], "group_name": "middle"},
        {"params": model.features[6:9].parameters(), "lr": lrs[2], "group_name": "late"},
        {"params": head_params, "lr": lrs[3], "group_name": "head"},
    ]

    if neck_params:
        groups.append({
            "params": neck_params,
            "lr": base_lr * float(neck_lr_ratio),
            "group_name": "neck",
        })

    fusion_params = []
    for name in ("fusion", "pooled_fusion"):
        fusion = getattr(model, name, None)
        if fusion is not None:
            fusion_params += list(fusion.parameters())

    if fusion_params:
        groups.append({
            "params": fusion_params,
            "lr": base_lr * fusion_lr_ratio,
            "group_name": "fusion",
        })

    return groups


def build_optimizer(model, config):
    optimizer_config = config["optimizer"]
    optimizer_name = optimizer_config["name"].lower()

    if optimizer_name != "adamw":
        raise ValueError(f"Unsupported optimizer: {optimizer_config['name']}")

    weight_decay = float(optimizer_config["weight_decay"])

    if "base_lr" in optimizer_config:
        base_lr = float(optimizer_config["base_lr"])
        lr_ratio = optimizer_config["lr_ratio"]
        neck_lr_ratio = optimizer_config.get("neck_lr_ratio")

        parameter_groups = _build_discriminative_groups(
            model=model,
            base_lr=base_lr,
            lr_ratio=lr_ratio,
            fusion_lr_ratio=float(optimizer_config.get("fusion_lr_ratio", 3.0)),
            neck_lr_ratio=float(neck_lr_ratio) if neck_lr_ratio is not None else None,
        )

        return torch.optim.AdamW(parameter_groups, weight_decay=weight_decay)

    return torch.optim.AdamW(
        model.parameters(),
        lr=float(optimizer_config["lr"]),
        weight_decay=weight_decay,
    )