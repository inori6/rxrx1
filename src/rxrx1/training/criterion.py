import math

import torch
import torch.nn as nn

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

    def forward(self, logits, targets, embeddings=None, batch=None):
        loss = self.ce(logits, targets)
        # Two-argument interface remains CE-only for validation/inference callers.
        if embeddings is not None and self.metric_enabled:
            if batch is None or targets.ndim != 1:
                raise ValueError(
                    "Metric supervision requires original hard labels and sample metadata."
                )
            loss = loss + self.lambda_metric * self.metric(
                embeddings,
                targets,
                batch["cell_type_idx"].to(logits.device),
                batch["sample_idx"].to(logits.device),
            )
        return loss


def build_criterion(config):
    loss_config = config["loss"]

    if loss_config["name"].lower() == "cross_entropy":
        metric = config.get("metric") or {}
        if metric.get("enabled", False):
            return ClassificationMetricLoss(
                lambda_metric=metric.get("lambda_metric", 0.1),
                **{key: metric[key] for key in ("wt", "wc", "alpha", "min_pairs") if key in metric},
            )
        return nn.CrossEntropyLoss()

    raise ValueError(f"Unsupported loss: {loss_config['name']}")
