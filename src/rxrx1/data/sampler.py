"""Treatment-balanced batches of P distinct siRNAs and K observations each."""

import math

import torch
from torch.utils.data import Sampler


class PKBatchSampler(Sampler):
    def __init__(self, labels, batch_size, k=4, seed=0):
        if not isinstance(k, int) or k < 2 or not isinstance(batch_size, int) or batch_size % k:
            raise ValueError("PK sampling requires integer K >= 2 and batch_size divisible by K.")
        self.p, self.k = batch_size // k, k
        self.groups = {}
        for index, label in enumerate(labels):
            self.groups.setdefault(label, []).append(index)
        if not 1 <= self.p <= len(self.groups):
            raise ValueError("P must be between 1 and the number of distinct siRNAs.")
        self.groups = list(self.groups.values())
        self.num_batches = math.ceil(sum(map(len, self.groups)) / batch_size)
        self.seed, self.epoch = seed, 0

    def __len__(self):
        return self.num_batches

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __iter__(self):
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        for _ in range(len(self)):
            batch = []
            for group_idx in torch.randperm(len(self.groups), generator=generator)[
                : self.p
            ].tolist():
                group = self.groups[group_idx]
                # Use distinct observations first; replace only if a treatment has fewer than K.
                indices = torch.randperm(len(group), generator=generator)[: self.k].tolist()
                if len(indices) < self.k:
                    indices += torch.randint(
                        len(group), (self.k - len(indices),), generator=generator
                    ).tolist()
                batch.extend(group[i] for i in indices)
            yield batch
