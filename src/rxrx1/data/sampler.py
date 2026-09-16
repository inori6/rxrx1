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



class MixedPKBatchSampler(Sampler):
    """
    PK sampler for pseudo-label training.

    For every selected treatment/class:
      - if pseudo samples exist: real_k real + pseudo_k pseudo
      - otherwise: k real samples

    Example:
      batch_size=32, k=4, pseudo_k=1
      -> P=8 classes
      -> normally 3 real + 1 pseudo per class
      -> maximum 24 real + 8 pseudo per batch

    Sampling is random every epoch. Confidence never determines
    within-epoch sample order.
    """

    def __init__(
        self,
        labels,
        is_pseudo,
        batch_size,
        k=4,
        pseudo_k=1,
        seed=0,
    ):
        if not isinstance(k,int) or k < 2:
            raise ValueError("k must be an integer >= 2.")
        if not isinstance(pseudo_k,int) or not 0 <= pseudo_k < k:
            raise ValueError("pseudo_k must satisfy 0 <= pseudo_k < k.")
        if not isinstance(batch_size,int) or batch_size % k:
            raise ValueError("batch_size must be divisible by k.")
        if len(labels) != len(is_pseudo):
            raise ValueError("labels and is_pseudo must have the same length.")

        self.p=batch_size//k
        self.k=k
        self.pseudo_k=pseudo_k
        self.real_k=k-pseudo_k
        self.seed=seed
        self.epoch=0

        self.real_groups={}
        self.pseudo_groups={}

        for index,(label,pseudo) in enumerate(zip(labels,is_pseudo)):
            groups=self.pseudo_groups if bool(pseudo) else self.real_groups
            groups.setdefault(label,[]).append(index)

        # Only classes backed by real labels may participate.
        self.labels=sorted(self.real_groups)

        if not 1 <= self.p <= len(self.labels):
            raise ValueError(
                "P must be between 1 and the number of real-label classes."
            )

        total_real=sum(len(v) for v in self.real_groups.values())
        real_per_batch=self.p*self.real_k

        if real_per_batch <= 0:
            raise ValueError("Each batch must contain real samples.")

        # Define one epoch primarily by the amount of REAL training data.
        self.num_batches=math.ceil(total_real/real_per_batch)

    def __len__(self):
        return self.num_batches

    def set_epoch(self,epoch):
        self.epoch=epoch

    @staticmethod
    def _sample(group,n,generator):
        if n <= 0:
            return []

        if len(group) >= n:
            local=torch.randperm(
                len(group),
                generator=generator,
            )[:n].tolist()
            return [group[i] for i in local]

        # Use every available observation once before replacement.
        local=torch.randperm(
            len(group),
            generator=generator,
        ).tolist()

        result=[group[i] for i in local]

        extra=torch.randint(
            len(group),
            (n-len(result),),
            generator=generator,
        ).tolist()

        result.extend(group[i] for i in extra)
        return result

    def __iter__(self):
        generator=torch.Generator().manual_seed(
            self.seed+self.epoch
        )

        for _ in range(len(self)):
            batch=[]

            chosen=torch.randperm(
                len(self.labels),
                generator=generator,
            )[:self.p].tolist()

            for label_idx in chosen:
                label=self.labels[label_idx]

                real_group=self.real_groups[label]
                pseudo_group=self.pseudo_groups.get(label,[])

                if pseudo_group:
                    real_n=self.real_k
                    pseudo_n=self.pseudo_k
                else:
                    # No valid pseudo for this class:
                    # fill the whole K with real samples.
                    real_n=self.k
                    pseudo_n=0

                batch.extend(
                    self._sample(
                        real_group,
                        real_n,
                        generator,
                    )
                )

                if pseudo_n:
                    batch.extend(
                        self._sample(
                            pseudo_group,
                            pseudo_n,
                            generator,
                        )
                    )

            # Randomize positions inside the batch as well.
            order=torch.randperm(
                len(batch),
                generator=generator,
            ).tolist()

            yield [batch[i] for i in order]
