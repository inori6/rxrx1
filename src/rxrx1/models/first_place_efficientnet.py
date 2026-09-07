import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import EfficientNet_B2_Weights, efficientnet_b2

from rxrx1.models.efficientnet import _replace_input_conv
from rxrx1.models.metadata import MetadataFusion
from rxrx1.models.rcic1st_common import (
    ArcMarginProduct,
    ChampionNeck,
    set_batch_norm_momentum,
)


class FirstPlaceEfficientNetB2(nn.Module):
    """First-place RxRx1 head/ArcFace recipe with EfficientNet-B2 backbone."""

    def __init__(
        self,
        num_classes=1108,
        pretrained=True,
        embedding_size=1024,
        bn_momentum=0.05,
        num_cell_types=4,
    ):
        super().__init__()
        weights = EfficientNet_B2_Weights.DEFAULT if pretrained else None
        base = efficientnet_b2(weights=weights)

        # The first-place public implementation replaces the six-channel stem
        # with a fresh convolution. Reuse the repository helper but deliberately
        # skip RGB-weight copying while keeping the rest of the backbone pretrained.
        _replace_input_conv(base, pretrained=False)

        self.features = base.features
        self.avgpool = base.avgpool
        feature_dim = base.classifier[1].in_features

        self.fusion = MetadataFusion(
            feature_dim=feature_dim,
            method="concat",
            cell_type=True,
            well_position=False,
            num_cell_types=num_cell_types,
        )

        neck = ChampionNeck(
            self.fusion.out_dim,
            embedding_size=embedding_size,
            bn_momentum=bn_momentum,
        )
        head = nn.Linear(embedding_size, num_classes)
        arc_margin_product = ArcMarginProduct(embedding_size, num_classes)
        self.classifier = nn.ModuleList([neck, head, arc_margin_product])

        # Match the first-place code, which overwrites momentum on all BN layers.
        set_batch_norm_momentum(self, bn_momentum)

    @property
    def neck(self):
        return self.classifier[0]

    @property
    def head(self):
        return self.classifier[1]

    @property
    def arc_margin_product(self):
        return self.classifier[2]

    def embed(self, x, metadata):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fusion(x, metadata)
        return self.neck(x)

    def forward(self, x, metadata=None, return_arc_logits=False):
        if metadata is None:
            raise ValueError("cell-type metadata is required by the first-place model")
        embedding = self.embed(x, metadata)
        logits = self.head(embedding)
        if return_arc_logits:
            return logits, self.arc_margin_product(embedding)
        return logits


def build_first_place_efficientnet(
    num_classes=1108,
    pretrained=True,
    embedding_size=1024,
    bn_momentum=0.05,
    num_cell_types=4,
):
    return FirstPlaceEfficientNetB2(
        num_classes=num_classes,
        pretrained=pretrained,
        embedding_size=embedding_size,
        bn_momentum=bn_momentum,
        num_cell_types=num_cell_types,
    )
