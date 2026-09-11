import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from rxrx1.training.pairs import (
    all_pair_indices,
    predicted_distances,
    target_distances,
    validate_distance_weights,
)


class HierarchicalMetricLoss(nn.Module):
    def __init__(self, wt=0.7, wc=0.3, alpha=0.8, min_pairs=8):
        super().__init__()
        validate_distance_weights(wt, wc, alpha)
        if not math.isfinite(min_pairs) or min_pairs <= 0:
            raise ValueError("min_pairs must be positive and finite.")
        self.wt, self.wc, self.alpha, self.min_pairs = wt, wc, alpha, min_pairs

    def forward(self, z, labels, cell_type_idx, sample_idx):
        if z.ndim != 2 or len(z) != len(labels):
            raise ValueError("Embeddings must have shape [batch_size, projection_dim].")
        pairs = all_pair_indices(len(z), z.device)
        target, relation = target_distances(
            labels, cell_type_idx, sample_idx, pairs, self.wt, self.wc, self.alpha
        )
        errors = (predicted_distances(z, pairs) - target).square()
        counts = torch.bincount(relation, minlength=5).to(errors.dtype)
        sums = errors.new_zeros(5).scatter_add(0, relation, errors)
        means = sums / counts.clamp_min(1)
        weights = (counts / self.min_pairs).clamp_max(1)
        # Empty batches/pairs produce a differentiable zero.
        return (means * weights).sum() / weights.sum().clamp_min(torch.finfo(errors.dtype).eps)


class ClassificationMetricLoss(nn.Module):
    def __init__(self, lambda_metric=0.1, **kwargs):
        super().__init__()
        if not math.isfinite(lambda_metric) or lambda_metric < 0:
            raise ValueError("lambda_metric must be finite and nonnegative.")
        self.lambda_metric = lambda_metric
        self.metric_enabled = lambda_metric > 0
        self.ce = nn.CrossEntropyLoss()
        self.metric = HierarchicalMetricLoss(**kwargs)

    def classification_loss(self, logits, targets):
        if targets.ndim == 1:
            return F.cross_entropy(logits, targets)

        logprobs = F.log_softmax(logits.float(), dim=-1)

        return (
                -logprobs * targets.float()
        ).sum(dim=-1).mean()

    def metric_loss(self, embeddings, labels, batch):
        if labels.ndim != 1:
            raise ValueError("Metric labels must be hard class indices.")

        return self.metric(
            embeddings,
            labels,
            batch["cell_type_idx"].to(embeddings.device),
            batch["sample_idx"].to(embeddings.device),
        )

    def forward(self, logits, targets, embeddings=None, batch=None):
        loss = self.classification_loss(logits, targets)

        if embeddings is not None and self.metric_enabled:
            if batch is None or targets.ndim != 1:
                raise ValueError(
                    "Metric supervision requires original hard labels and sample metadata."
                )

            loss = loss + self.lambda_metric * self.metric_loss(
                embeddings,
                targets,
                batch,
            )

        return loss


class DenseCrossEntropy(nn.Module):
    def forward(self, logits, targets):
        if targets.ndim == 1:
            return F.cross_entropy(logits, targets)
        logprobs = F.log_softmax(logits.float(), dim=-1)
        return (-logprobs * targets.float()).sum(-1).mean()


class ArcFaceLoss(nn.Module):
    def __init__(self, s=30.0, m=0.5, divisor=2.0):
        super().__init__()
        if not math.isfinite(s) or s <= 0:
            raise ValueError("ArcFace scale must be positive and finite.")
        if not math.isfinite(m) or m <= 0:
            raise ValueError("ArcFace margin must be positive and finite.")
        if not math.isfinite(divisor) or divisor <= 0:
            raise ValueError("ArcFace divisor must be positive and finite.")
        self.s = s
        self.divisor = divisor
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m
        self.ce = DenseCrossEntropy()

    def forward(self, cosine, targets):
        cosine = cosine.float()
        if targets.ndim == 1:
            targets = F.one_hot(targets, num_classes=cosine.shape[1]).float()
        else:
            targets = targets.float()

        sine = torch.sqrt((1.0 - cosine.square()).clamp_min(0.0))
        phi = cosine * self.cos_m - sine * self.sin_m
        phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        output = (targets * phi) + ((1.0 - targets) * cosine)
        return self.ce(output * self.s, targets) / self.divisor


class ClassificationArcFaceLoss(nn.Module):
    arcface_enabled = True

    def __init__(self, metric_loss_coeff=0.2, s=30.0, m=0.5, divisor=2.0):
        super().__init__()
        if not math.isfinite(metric_loss_coeff) or not 0 <= metric_loss_coeff <= 1:
            raise ValueError("metric_loss_coeff must be finite and between 0 and 1.")
        self.metric_loss_coeff = metric_loss_coeff
        self.ce = DenseCrossEntropy()
        self.arcface = ArcFaceLoss(s=s, m=m, divisor=divisor)

    def forward(self, logits, targets, arc_logits=None):
        ce_loss = self.ce(logits, targets)
        # Validation/inference keeps the public first-place classification path:
        # raw embedding -> independent FC head, with no ArcFace branch required.
        if arc_logits is None:
            return ce_loss
        metric_loss = self.arcface(arc_logits, targets)
        coeff = self.metric_loss_coeff
        return ce_loss * (1.0 - coeff) + metric_loss * coeff


def build_criterion(config):
    loss_config = config["loss"]
    loss_name = loss_config["name"].lower()

    if loss_name == "cross_entropy":
        metric = config.get("metric") or {}
        if metric.get("enabled", False):
            return ClassificationMetricLoss(
                lambda_metric=metric.get("lambda_metric", 0.1),
                **{key: metric[key] for key in ("wt", "wc", "alpha", "min_pairs") if key in metric},
            )
        return nn.CrossEntropyLoss()

    if loss_name == "first_place_arcface":
        return ClassificationArcFaceLoss(
            metric_loss_coeff=loss_config.get("metric_loss_coeff", 0.2),
            s=loss_config.get("arcface_scale", 30.0),
            m=loss_config.get("arcface_margin", 0.5),
            divisor=loss_config.get("arcface_divisor", 2.0),
        )

    raise ValueError(f"Unsupported loss: {loss_config['name']}")
