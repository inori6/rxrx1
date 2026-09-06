import torch
import torch.nn as nn
from torchvision.models import (
    EfficientNet_B2_Weights,
    EfficientNet_B4_Weights,
    efficientnet_b2,
    efficientnet_b4,
)

from rxrx1.models.metadata import MetadataFusion


_BUILDERS = {
    "efficientnet_b2": efficientnet_b2,
    "efficientnet_b4": efficientnet_b4,
}

_WEIGHTS = {
    "efficientnet_b2": EfficientNet_B2_Weights.DEFAULT,
    "efficientnet_b4": EfficientNet_B4_Weights.DEFAULT,
}


def _replace_input_conv(model, pretrained=True):
    old_conv = model.features[0][0]
    new_conv = nn.Conv2d(
        6,
        old_conv.out_channels,
        kernel_size=old_conv.kernel_size,
        stride=old_conv.stride,
        padding=old_conv.padding,
        bias=False,
    )

    if pretrained:
        with torch.no_grad():
            new_conv.weight.copy_(old_conv.weight.repeat(1, 2, 1, 1) / 2)

    model.features[0][0] = new_conv


class EfficientNetWithMetadata(nn.Module):
    def __init__(
        self,
        name="efficientnet_b2",
        num_classes=1108,
        pretrained=True,
        dropout=None,
        metadata=None,
    ):
        super().__init__()

        if name not in _BUILDERS:
            raise ValueError(f"Unsupported EfficientNet: {name}")

        weights = _WEIGHTS[name] if pretrained else None
        base_model = _BUILDERS[name](weights=weights)

        _replace_input_conv(base_model, pretrained)

        if dropout is not None:
            dropout = float(dropout)
            if not 0 <= dropout < 1:
                raise ValueError(
                    f"dropout must satisfy 0 <= dropout < 1, got {dropout}"
                )
            base_model.classifier[0].p = dropout

        self.features = base_model.features
        self.avgpool = base_model.avgpool

        feature_dim = base_model.classifier[1].in_features

        metadata = metadata or {}
        metadata_enabled = bool(metadata.get("enabled", False))

        if metadata_enabled:
            self.fusion = MetadataFusion(
                feature_dim=feature_dim,
                method=metadata.get("method", "concat"),
                cell_type=metadata.get("cell_type", False),
                well_position=metadata.get("well_position", False),
                num_cell_types=metadata.get("num_cell_types", 4),
                well_dim=metadata.get("well_dim", 2),
            )
            classifier_in = self.fusion.out_dim
        else:
            self.fusion = None
            classifier_in = feature_dim

        self.classifier = nn.Sequential(
            base_model.classifier[0],
            nn.Linear(classifier_in, num_classes),
        )

    def forward(self, x, metadata=None):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        if self.fusion is not None:
            if metadata is None:
                raise ValueError(
                    "metadata is required when metadata fusion is enabled"
                )
            x = self.fusion(x, metadata)

        return self.classifier(x)


def build_efficientnet(
    name="efficientnet_b2",
    num_classes=1108,
    pretrained=True,
    dropout=None,
    metadata=None,
):
    return EfficientNetWithMetadata(
        name=name,
        num_classes=num_classes,
        pretrained=pretrained,
        dropout=dropout,
        metadata=metadata,
    )


if __name__ == "__main__":
    x = torch.randn(2, 6, 128, 128)

    metadata = {
        "cell_type_idx": torch.tensor([0, 3]),
        "well_position": torch.tensor([
            [0.0, 0.0],
            [1.0, 1.0],
        ]),
    }

    model = build_efficientnet(
        pretrained=False,
        metadata={
            "enabled": True,
            "method": "concat",
            "cell_type": True,
            "well_position": True,
        },
    )

    output = model(x, metadata)

    print("output:", output.shape)
    print("classifier:", model.classifier)