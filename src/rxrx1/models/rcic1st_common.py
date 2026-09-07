import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class ChampionNeck(nn.Sequential):
    def __init__(self, in_features, embedding_size=1024, bn_momentum=0.05):
        super().__init__(
            nn.BatchNorm1d(in_features, momentum=bn_momentum),
            nn.Linear(in_features, embedding_size, bias=False),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(embedding_size, momentum=bn_momentum),
            nn.Linear(embedding_size, embedding_size, bias=False),
            nn.BatchNorm1d(embedding_size, momentum=bn_momentum),
        )


class ArcMarginProduct(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.weight.size(1))
        nn.init.uniform_(self.weight, -stdv, stdv)

    def forward(self, features):
        return F.linear(F.normalize(features), F.normalize(self.weight))


def set_batch_norm_momentum(module, momentum=0.05):
    for child in module.modules():
        if isinstance(child, (nn.BatchNorm1d, nn.BatchNorm2d)):
            child.momentum = momentum
