import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import (
    EfficientNet_B2_Weights,
    EfficientNet_B4_Weights,
    efficientnet_b2,
    efficientnet_b4,
)

from rxrx1.models.efficientnet import _replace_input_conv
from rxrx1.models.metadata import MetadataFusion
from rxrx1.models.rcic1st_common import ChampionNeck


_BUILDERS = {
    "efficientnet_b2_metric_neck": efficientnet_b2,
    "efficientnet_b4_metric_neck": efficientnet_b4,
}

_WEIGHTS = {
    "efficientnet_b2_metric_neck": EfficientNet_B2_Weights.DEFAULT,
    "efficientnet_b4_metric_neck": EfficientNet_B4_Weights.DEFAULT,
}


class EfficientNetMetricNeck(nn.Module):
    def __init__(
        self,
        name="efficientnet_b2_metric_neck",
        num_classes=1108,
        pretrained=True,
        dropout=0.22,
        embedding_size=1024,
        bn_momentum=0.05,
        neck_layers=1,
        metric=None,
        metadata=None,
    ):
        super().__init__()

        if name not in _BUILDERS:
            raise ValueError(f"Unsupported metric-neck model: {name}")

        weights = _WEIGHTS[name] if pretrained else None
        base = _BUILDERS[name](weights=weights)

        _replace_input_conv(base, pretrained=pretrained)

        self.features = base.features
        self.avgpool = base.avgpool
        feature_dim = base.classifier[1].in_features

        dropout_layer = base.classifier[0]
        dropout_layer.p = float(dropout)
        dropout_layer.inplace = False

        metadata = metadata or {}
        self.fusion = None

        if metadata.get("enabled", False):
            if metadata.get("method", "concat") != "concat":
                raise ValueError(
                    "Metric neck metadata only supports concat."
                )

            self.fusion = MetadataFusion(
                feature_dim=feature_dim,
                method="concat",
                cell_type=metadata.get("cell_type", False),
                well_position=metadata.get("well_position", False),
                num_cell_types=metadata.get("num_cell_types", 4),
                well_dim=metadata.get("well_dim", 2),
            )

            feature_dim = self.fusion.out_dim

        self.classifier = nn.ModuleList([
            ChampionNeck(
                feature_dim,
                embedding_size=embedding_size,
                bn_momentum=bn_momentum,
                neck_layers=neck_layers,
            ),
            dropout_layer,
            nn.Linear(embedding_size, num_classes),
        ])

        metric = metric or {}
        self.projection = None

        if metric.get("enabled", False):
            dims = metric.get(
                "projection_dims",
                [512, 128],
            )

            if not dims or any(
                not isinstance(dim, int) or dim <= 0
                for dim in dims
            ):
                raise ValueError(
                    "projection_dims must contain positive integers."
                )

            # Do not disturb classifier RNG state.
            with torch.random.fork_rng(devices=[]):
                layers = []
                in_dim = embedding_size

                for index, out_dim in enumerate(dims):
                    layers.append(
                        nn.Linear(in_dim, out_dim)
                    )

                    if index < len(dims) - 1:
                        layers.extend([
                            nn.BatchNorm1d(out_dim),
                            nn.SiLU(),
                        ])

                    in_dim = out_dim

                self.projection = nn.Sequential(*layers)

    @property
    def neck(self):
        return self.classifier[0]

    @property
    def dropout(self):
        return self.classifier[1]

    @property
    def head(self):
        return self.classifier[2]

    def forward(
        self,
        x,
        metadata=None,
        return_embeddings=False,
    ):
        # EfficientNet
        x = self.features(x)

        # GAP
        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        # cell type + well position
        if self.fusion is not None:
            if metadata is None:
                raise ValueError(
                    "metadata is required when concat is enabled"
                )

            x = self.fusion(x, metadata)

        # single linear neck
        x = self.neck(x)

        # metric branch
        if return_embeddings:
            if self.projection is None:
                raise ValueError(
                    "Enable metric projection before "
                    "requesting embeddings."
                )

            z = F.normalize(
                self.projection(x),
                dim=1,
            )

        # classification branch
        logits = self.head(
            self.dropout(x)
        )

        if return_embeddings:
            return logits, z

        return logits


def build_efficientnet_metric_neck(
    name="efficientnet_b2_metric_neck",
    num_classes=1108,
    pretrained=True,
    dropout=0.22,
    embedding_size=1024,
    bn_momentum=0.05,
    neck_layers=1,
    metric=None,
    metadata=None,
):
    return EfficientNetMetricNeck(
        name=name,
        num_classes=num_classes,
        pretrained=pretrained,
        dropout=dropout,
        embedding_size=embedding_size,
        bn_momentum=bn_momentum,
        neck_layers=neck_layers,
        metric=metric,
        metadata=metadata,
    )