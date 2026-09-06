"""Unordered observation pairs; relation IDs follow the hierarchy below."""

import math

import torch
import torch.nn.functional as F

RELATIONS = ("same_sample", "same_t_same_c", "same_t_diff_c", "diff_t_same_c", "diff_t_diff_c")


def all_pair_indices(batch_size, device=None):
    """Return [2, B*(B-1)//2] indices, without diagonal or reverse duplicates."""
    return torch.triu_indices(batch_size, batch_size, offset=1, device=device)


def validate_distance_weights(wt, wc, alpha):
    if not (0 <= wt <= 1 and 0 <= wc <= 1 and math.isclose(wt + wc, 1, abs_tol=1e-6)):
        raise ValueError("wt and wc must be in [0, 1] and sum to 1.")
    if not 0 <= alpha <= 1:
        raise ValueError("alpha must be in [0, 1].")


def target_distances(labels, cell_type_idx, sample_idx, pairs=None, wt=0.7, wc=0.3, alpha=0.8):
    validate_distance_weights(wt, wc, alpha)
    if labels.ndim != 1 or cell_type_idx.shape != labels.shape or sample_idx.shape != labels.shape:
        raise ValueError("Pair metadata must be matching one-dimensional hard-label tensors.")
    if pairs is None:
        pairs = all_pair_indices(len(labels), labels.device)
    i, j = pairs
    same_t = labels[i] == labels[j]
    same_c = cell_type_idx[i] == cell_type_idx[j]
    same_s = sample_idx[i] == sample_idx[j]
    if (same_s & ~(same_t & same_c)).any():
        raise ValueError("The same sample must have consistent treatment and cell type.")
    similarity = wt * same_t.float() + wc * same_c.float()
    distance = 1 - torch.where(same_s, similarity, alpha * similarity)
    relation = torch.where(same_t, torch.where(same_c, 1, 2), torch.where(same_c, 3, 4))
    return distance, relation.masked_fill(same_s, 0)


def predicted_distances(z, pairs=None):
    if pairs is None:
        pairs = all_pair_indices(len(z), z.device)
    z = F.normalize(z.float(), dim=1)
    i, j = pairs
    return (1 - (z[i] * z[j]).sum(dim=1).clamp(-1, 1)) / 2
