import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import EfficientNet_B2_Weights, efficientnet_b2

from rxrx1.models.efficientnet import _replace_input_conv
from rxrx1.models.rcic1st_common import ChampionNeck


class EfficientNetB2MetricNeck(nn.Module):
    def __init__(
        self,
        num_classes=1108,
        pretrained=True,
        dropout=0.22,
        embedding_size=1024,
        bn_momentum=0.05,
        metric=None,
    ):
        super().__init__()

        weights = EfficientNet_B2_Weights.DEFAULT if pretrained else None
        base = efficientnet_b2(weights=weights)
        _replace_input_conv(base, pretrained=pretrained)

        self.features = base.features
        self.avgpool = base.avgpool
        feature_dim = base.classifier[1].in_features

        dropout_layer = base.classifier[0]
        dropout_layer.p = float(dropout)
        dropout_layer.inplace = False

        self.classifier = nn.ModuleList([
            ChampionNeck(
                feature_dim,
                embedding_size=embedding_size,
                bn_momentum=bn_momentum,
            ),
            dropout_layer,
            nn.Linear(embedding_size, num_classes),
        ])

        metric = metric or {}
        self.projection = None

        if metric.get("enabled", False):
            dims = metric.get("projection_dims", [512, 128])

            if not dims or any(
                not isinstance(dim, int) or dim <= 0
                for dim in dims
            ):
                raise ValueError(
                    "projection_dims must contain positive integers."
                )

            with torch.random.fork_rng(devices=[]):
                layers = []
                in_dim = embedding_size

                for index, out_dim in enumerate(dims):
                    layers.append(nn.Linear(in_dim, out_dim))

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
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        x = self.neck(x)

        if return_embeddings:
            if self.projection is None:
                raise ValueError(
                    "Enable metric projection before requesting embeddings."
                )

            z = F.normalize(
                self.projection(x),
                dim=1,
            )

        logits = self.head(
            self.dropout(x)
        )

        if return_embeddings:
            return logits, z

        return logits


def build_efficientnet_metric_neck(
    num_classes=1108,
    pretrained=True,
    dropout=0.22,
    embedding_size=1024,
    bn_momentum=0.05,
    metric=None,
):
    return EfficientNetB2MetricNeck(
        num_classes=num_classes,
        pretrained=pretrained,
        dropout=dropout,
        embedding_size=embedding_size,
        bn_momentum=bn_momentum,
        metric=metric,
    )