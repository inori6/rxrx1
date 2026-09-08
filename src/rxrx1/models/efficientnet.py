import torch
import torch.nn as nn
import torch.nn.functional as F
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
        pooled_metadata=None,
        metric=None,
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
                raise ValueError(f"dropout must satisfy 0 <= dropout < 1, got {dropout}")
            base_model.classifier[0].p = dropout

        base_model.classifier[0].inplace = False
        self.features = base_model.features
        self.avgpool = base_model.avgpool
        feature_dim = base_model.classifier[1].in_features

        metric = metric or {}
        self.projection = None

        if metric.get("enabled", False):
            dims = metric.get("projection_dims", [512, 128])

            if not dims or any(not isinstance(d, int) or d <= 0 for d in dims):
                raise ValueError("projection_dims must contain positive integers.")

            with torch.random.fork_rng(devices=[]):
                layers = []
                in_dim = feature_dim

                for i, out_dim in enumerate(dims):
                    layers.append(nn.Linear(in_dim, out_dim))

                    if i < len(dims) - 1:
                        layers += [
                            nn.BatchNorm1d(out_dim),
                            nn.SiLU(),
                        ]

                    in_dim = out_dim

                self.projection = nn.Sequential(*layers)

        metadata = metadata or {}
        metadata_enabled = bool(metadata.get("enabled", False))
        self.fusion_feature_index = metadata.get("feature_index")
        fusion_dim = feature_dim

        if self.fusion_feature_index is not None:
            if not metadata_enabled or metadata.get("method", "concat") != "film":
                raise ValueError("metadata.feature_index requires enabled FiLM fusion.")

            index = self.fusion_feature_index

            if type(index) is not int or not 1 <= index < len(self.features) - 1:
                raise ValueError("metadata.feature_index must select an MBConv stage (1–7).")

            fusion_dim = self.features[index][-1].out_channels

        if metadata_enabled:
            rng_state = torch.get_rng_state()

            self.fusion = MetadataFusion(
                feature_dim=fusion_dim,
                method=metadata.get("method", "concat"),
                cell_type=metadata.get("cell_type", False),
                well_position=metadata.get("well_position", False),
                num_cell_types=metadata.get("num_cell_types", 4),
                well_dim=metadata.get("well_dim", 2),
            )

            torch.set_rng_state(rng_state)

            classifier_in = (
                feature_dim
                if self.fusion_feature_index is not None
                else self.fusion.out_dim
            )
        else:
            self.fusion = None
            classifier_in = feature_dim

        pooled_metadata = pooled_metadata or {}

        if pooled_metadata.get("enabled", False):
            rng_state = torch.get_rng_state()

            self.pooled_fusion = MetadataFusion(
                feature_dim=classifier_in,
                method=pooled_metadata.get("method", "concat"),
                cell_type=pooled_metadata.get("cell_type", False),
                well_position=pooled_metadata.get("well_position", False),
                num_cell_types=pooled_metadata.get("num_cell_types", 4),
                well_dim=pooled_metadata.get("well_dim", 2),
            )

            torch.set_rng_state(rng_state)
            classifier_in = self.pooled_fusion.out_dim
        else:
            self.pooled_fusion = None

        self.classifier = nn.Sequential(
            base_model.classifier[0],
            nn.Linear(classifier_in, num_classes),
        )

    def forward(self, x, metadata=None, return_embeddings=False):
        if (self.fusion is not None or self.pooled_fusion is not None) and metadata is None:
            raise ValueError("metadata is required when metadata fusion is enabled")

        if self.fusion_feature_index is None:
            x = self.features(x)
        else:
            for i, stage in enumerate(self.features):
                x = stage(x)

                if i == self.fusion_feature_index:
                    x = self.fusion(x, metadata)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        if return_embeddings:
            if self.projection is None:
                raise ValueError("Enable metric projection before requesting embeddings.")
            z = F.normalize(self.projection(x), dim=1)

        if self.fusion is not None and self.fusion_feature_index is None:
            x = self.fusion(x, metadata)

        if self.pooled_fusion is not None:
            x = self.pooled_fusion(x, metadata)

        logits = self.classifier(x)

        return (logits, z) if return_embeddings else logits


def build_efficientnet(
    name="efficientnet_b2",
    num_classes=1108,
    pretrained=True,
    dropout=None,
    metadata=None,
    pooled_metadata=None,
    metric=None,
):
    return EfficientNetWithMetadata(
        name=name,
        num_classes=num_classes,
        pretrained=pretrained,
        dropout=dropout,
        metadata=metadata,
        pooled_metadata=pooled_metadata,
        metric=metric,
    )


if __name__ == "__main__":
    x = torch.randn(2, 6, 128, 128)

    metadata = {
        "cell_type_idx": torch.tensor([0, 3]),
        "well_position": torch.tensor([[0.0, 0.0], [1.0, 1.0]]),
    }

    model = build_efficientnet(
        pretrained=False,
        metadata={
            "enabled": True,
            "method": "film",
            "feature_index": 2,
            "cell_type": False,
            "well_position": True,
        },
        pooled_metadata={
            "enabled": True,
            "method": "concat",
            "cell_type": True,
            "well_position": False,
        },
    )

    output = model(x, metadata)

    print("output:", output.shape)
    print("classifier:", model.classifier)