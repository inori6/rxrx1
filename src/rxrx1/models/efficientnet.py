import torch
import torch.nn as nn
from torchvision.models import (
    EfficientNet_B2_Weights,
    EfficientNet_B4_Weights,
    efficientnet_b2,
    efficientnet_b4,
)


_BUILDERS = {
    "efficientnet_b2": efficientnet_b2,
    "efficientnet_b4": efficientnet_b4,
}

_WEIGHTS = {
    "efficientnet_b2": (
        EfficientNet_B2_Weights.DEFAULT
    ),
    "efficientnet_b4": (
        EfficientNet_B4_Weights.DEFAULT
    ),
}


def _replace_input_conv(
    model,
    pretrained=True,
):
    old_conv = model.features[0][0]

    new_conv = nn.Conv2d(
        in_channels=6,
        out_channels=old_conv.out_channels,
        kernel_size=old_conv.kernel_size,
        stride=old_conv.stride,
        padding=old_conv.padding,
        bias=False,
    )

    if pretrained:
        with torch.no_grad():
            new_conv.weight.copy_(
                old_conv.weight.repeat(
                    1,
                    2,
                    1,
                    1,
                )
                / 2
            )

    model.features[0][0] = new_conv


def build_efficientnet(
    name="efficientnet_b2",
    num_classes=1108,
    pretrained=True,
    dropout=None,
):
    if name not in _BUILDERS:
        raise ValueError(
            f"Unsupported EfficientNet: {name}"
        )

    weights = (
        _WEIGHTS[name]
        if pretrained
        else None
    )

    model = _BUILDERS[name](
        weights=weights
    )

    _replace_input_conv(
        model,
        pretrained,
    )

    if dropout is not None:
        dropout = float(dropout)

        if not 0 <= dropout < 1:
            raise ValueError(
                "dropout must satisfy "
                f"0 <= dropout < 1, got {dropout}"
            )

        model.classifier[0].p = dropout

    model.classifier[1] = nn.Linear(
        model.classifier[1].in_features,
        num_classes,
    )

    return model


if __name__ == "__main__":
    model_b2 = build_efficientnet(
        "efficientnet_b2"
    )
    print(model_b2)

    model_b4 = build_efficientnet(
        "efficientnet_b4"
    )
    print(model_b4)