import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import DenseNet161_Weights, densenet161

from rxrx1.models.metadata import MetadataFusion
from rxrx1.models.rcic1st_common import (
    ArcMarginProduct,
    ChampionNeck,
    set_batch_norm_momentum,
)


class FirstPlaceDenseNet161(nn.Module):
    """RxRx1 first-place DenseNet161 classification path with ArcFace branch."""

    def __init__(
        self,
        num_classes=1108,
        pretrained=True,
        embedding_size=1024,
        bn_momentum=0.05,
        num_cell_types=4,
    ):
        super().__init__()
        weights = DenseNet161_Weights.DEFAULT if pretrained else None
        base = densenet161(weights=weights, memory_efficient=True)

        # Match the public first-place code: replace the ImageNet conv with a
        # freshly initialized six-channel convolution rather than copying RGB weights.
        base.features.conv0 = nn.Conv2d(6, 96, kernel_size=7, stride=2, padding=3, bias=False)

        # Keep exactly nine optimizer-visible groups so the repository's existing
        # EfficientNet discriminative-LR helper can be reused without special cases.
        self.features = nn.Sequential(
            nn.Sequential(
                base.features.conv0,
                base.features.norm0,
                base.features.relu0,
                base.features.pool0,
            ),
            base.features.denseblock1,
            base.features.transition1,
            base.features.denseblock2,
            base.features.transition2,
            base.features.denseblock3,
            base.features.transition3,
            base.features.denseblock4,
            base.features.norm5,
        )

        feature_dim = base.classifier.in_features
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

        # Existing optimizer code treats model.classifier as the head group.
        self.classifier = nn.ModuleList([neck, head, arc_margin_product])
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
        x = F.adaptive_avg_pool2d(x, (1, 1))
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


def build_first_place_densenet(
    num_classes=1108,
    pretrained=True,
    embedding_size=1024,
    bn_momentum=0.05,
    num_cell_types=4,
):
    return FirstPlaceDenseNet161(
        num_classes=num_classes,
        pretrained=pretrained,
        embedding_size=embedding_size,
        bn_momentum=bn_momentum,
        num_cell_types=num_cell_types,
    )
