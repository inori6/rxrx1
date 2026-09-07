import torch
import torch.nn as nn
from torchvision.models import EfficientNet_B2_Weights, efficientnet_b2

from rxrx1.models.efficientnet import _replace_input_conv
from rxrx1.models.metadata import MetadataFusion
from rxrx1.models.rcic1st_common import ChampionNeck


class EfficientNetB2ConcatNeck(nn.Module):
    """EfficientNet-B2 + optional metadata concat + first-place neck + CE head."""

    def __init__(
        self,
        num_classes=1108,
        pretrained=True,
        embedding_size=1024,
        bn_momentum=0.05,
        metadata=None,
    ):
        super().__init__()
        weights = EfficientNet_B2_Weights.DEFAULT if pretrained else None
        base = efficientnet_b2(weights=weights)

        # Keep the repository baseline's ImageNet-preserving six-channel stem.
        _replace_input_conv(base, pretrained=pretrained)

        self.features = base.features
        self.avgpool = base.avgpool
        feature_dim = base.classifier[1].in_features

        metadata = metadata or {}
        if metadata.get("enabled", False):
            if str(metadata.get("method", "concat")).lower() != "concat":
                raise ValueError("EfficientNetB2ConcatNeck only supports metadata method='concat'.")
            self.fusion = MetadataFusion(
                feature_dim=feature_dim,
                method="concat",
                cell_type=metadata.get("cell_type", False),
                well_position=metadata.get("well_position", False),
                num_cell_types=metadata.get("num_cell_types", 4),
                well_dim=metadata.get("well_dim", 2),
            )
            classifier_in = self.fusion.out_dim
        else:
            self.fusion = None
            classifier_in = feature_dim

        neck = ChampionNeck(
            classifier_in,
            embedding_size=embedding_size,
            bn_momentum=bn_momentum,
        )
        head = nn.Linear(embedding_size, num_classes)
        self.classifier = nn.ModuleList([neck, head])

    @property
    def neck(self):
        return self.classifier[0]

    @property
    def head(self):
        return self.classifier[1]

    def forward(self, x, metadata=None):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        if self.fusion is not None:
            if metadata is None:
                raise ValueError("metadata is required when metadata concat is enabled")
            x = self.fusion(x, metadata)

        x = self.neck(x)
        return self.head(x)


def build_efficientnet_concat_neck(
    num_classes=1108,
    pretrained=True,
    embedding_size=1024,
    bn_momentum=0.05,
    metadata=None,
):
    return EfficientNetB2ConcatNeck(
        num_classes=num_classes,
        pretrained=pretrained,
        embedding_size=embedding_size,
        bn_momentum=bn_momentum,
        metadata=metadata,
    )
