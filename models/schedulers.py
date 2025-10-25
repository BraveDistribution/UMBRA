"""Custom learning rate schedulers"""

from typing import Union, Sequence

import math
import torch
from torch.optim.lr_scheduler import _LRScheduler

from utils.misc import ensure_tuple_dim


class CosineAnnealingWithWarmup(_LRScheduler):
    """
    Cosine annealing with warmup scheduler.

    Adapted from https://github.com/MaastrichtU-CDS/anyBrainer/.
    """
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_iters: Union[int, Sequence[int]],
        total_iters: Union[int, Sequence[int]],
        start_iter: Union[int, Sequence[int]] = 0,
        eta_min: Union[float, Sequence[float]] = 1e-6,
        last_epoch: int = -1,
    ):
        """
        Args:
            optimizer: PyTorch optimizer.
            warmup_iters: Number of warmup iterations.
            total_iters: Total number of iterations (warmup + cosine decay).
            eta_min: Minimum LR after cosine decay.
            start_iter: The iteration to start the scheduler.
            last_epoch: The index of last epoch (or -1 if starting).
        """
        n_groups = len(optimizer.param_groups)

        self.warmup_iters = ensure_tuple_dim(warmup_iters, n_groups)
        self.total_iters = ensure_tuple_dim(total_iters, n_groups)
        self.eta_min = ensure_tuple_dim(eta_min, n_groups)
        self.start_iter = ensure_tuple_dim(start_iter, n_groups)

        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        lrs = []
        for i, base_lr in enumerate(self.base_lrs):
            start_iter = self.start_iter[i]
            warmup = self.warmup_iters[i]
            eta_min = self.eta_min[i]
            total_iters = self.total_iters[i]

            if self.last_epoch < start_iter:
                # Phase 1: Inactive
                lr = 0.0
            else:
                # Effective step after start
                effective_epoch = self.last_epoch - start_iter
                if effective_epoch < warmup:
                    # Phase 2: Linear warm-up
                    lr = base_lr * (effective_epoch + 1) / float(warmup)
                else:
                    # Phase 3: Cosine decay
                    decay_iters = total_iters - start_iter - warmup
                    progress = (effective_epoch - warmup) / float(decay_iters)
                    progress = min(max(progress, 0.0), 1.0)  # clamp to [0, 1]
                    lr = eta_min + (base_lr - eta_min) * 0.5 * (1.0 + math.cos(math.pi * progress))

            lrs.append(lr)

        return lrs