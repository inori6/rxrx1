import torch


_LR_RATIOS = {
    "1_1_3_10": (
        1.0,
        1.0,
        3.0,
        10.0,
    ),
    "1_3_5_10": (
        1.0,
        3.0,
        5.0,
        10.0,
    ),
    "1_3_10_30": (
        1.0,
        3.0,
        10.0,
        30.0,
    ),
}


def _parse_lr_ratio(value):
    if isinstance(value, str):
        if value not in _LR_RATIOS:
            raise ValueError(
                f"Unsupported lr_ratio: {value}"
            )

        return _LR_RATIOS[value]

    if isinstance(
        value,
        (list, tuple),
    ):
        if len(value) != 4:
            raise ValueError(
                "lr_ratio must contain "
                "exactly 4 values."
            )

        ratio = tuple(
            float(item)
            for item in value
        )

        if any(
            item <= 0
            for item in ratio
        ):
            raise ValueError(
                "lr_ratio values must "
                "be greater than zero."
            )

        return ratio

    raise TypeError(
        "lr_ratio must be a string, "
        "list, or tuple."
    )


def _build_discriminative_groups(
    model,
    base_lr,
    lr_ratio,
):
    if not hasattr(model, "features"):
        raise ValueError(
            "Discriminative learning rates "
            "require a model with "
            "model.features."
        )

    if not hasattr(model, "classifier"):
        raise ValueError(
            "Discriminative learning rates "
            "require model.classifier."
        )

    if len(model.features) < 9:
        raise ValueError(
            "Expected EfficientNet features "
            "to contain at least 9 stages."
        )

    ratios = _parse_lr_ratio(
        lr_ratio
    )

    lrs = [
        base_lr * ratio
        for ratio in ratios
    ]

    return [
        {
            "params": (
                model.features[0:3]
                .parameters()
            ),
            "lr": lrs[0],
            "group_name": "early",
        },
        {
            "params": (
                model.features[3:6]
                .parameters()
            ),
            "lr": lrs[1],
            "group_name": "middle",
        },
        {
            "params": (
                model.features[6:9]
                .parameters()
            ),
            "lr": lrs[2],
            "group_name": "late",
        },
        {
            "params": (
                model.classifier.parameters()
            ),
            "lr": lrs[3],
            "group_name": "head",
        },
    ]


def build_optimizer(
    model,
    config,
):
    optimizer_config = config["optimizer"]

    optimizer_name = (
        optimizer_config["name"].lower()
    )

    if optimizer_name != "adamw":
        raise ValueError(
            "Unsupported optimizer: "
            f"{optimizer_config['name']}"
        )

    weight_decay = float(
        optimizer_config[
            "weight_decay"
        ]
    )

    if "base_lr" in optimizer_config:
        base_lr = float(
            optimizer_config["base_lr"]
        )

        lr_ratio = optimizer_config[
            "lr_ratio"
        ]

        parameter_groups = (
            _build_discriminative_groups(
                model=model,
                base_lr=base_lr,
                lr_ratio=lr_ratio,
            )
        )

        return torch.optim.AdamW(
            parameter_groups,
            weight_decay=weight_decay,
        )

    return torch.optim.AdamW(
        model.parameters(),
        lr=float(
            optimizer_config["lr"]
        ),
        weight_decay=weight_decay,
    )