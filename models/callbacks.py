"""Custom Pytorch Lightning callbacks."""
from __future__ import annotations

__all__ = [
    "LogLR",
    "LogGradNorm",
]

from typing import Any

import lightning.pytorch as pl
import torch.optim as optim
from lightning.pytorch.utilities import rank_zero_only

from utils.nets import get_optimizer_lr, get_total_grad_norm


class LogLR(pl.Callback):
    """
    Callback to log the learning rate for all optimizers in the trainer.
    """
    @rank_zero_only
    def on_before_optimizer_step(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        optimizer: optim.Optimizer,
    ) -> None:
        """Log the learning rate."""
        if not isinstance(trainer.optimizers, list):
            print("Cannot log LR because pl_module.optimizers is not a list.")
            return
        
        pl_module.log_dict(
            get_optimizer_lr(trainer.optimizers),
            on_step=True,
            on_epoch=False,
            prog_bar=False,
            sync_dist=False,
            rank_zero_only=True,
        )


class LogGradNorm(pl.Callback):
    """
    Callback to log the gradient norm for all parameters in the model.
    """
    def on_train_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        """Log the gradient norm."""
        pl_module.log(
            "train/grad_norm",
            get_total_grad_norm(pl_module),
            on_step=True,
            on_epoch=False,
            prog_bar=False,
            sync_dist=True,
        )