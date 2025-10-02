"""
Foundation models for medical image analysis.

This module contains implementations of self-supervised pre-training models
and fine-tuning architectures for 3D medical image segmentation, including:
- ContrastiveTransformer: MoCo-based contrastive learning with masked autoencoding
- SegmentationFineTuner: Parameter-efficient fine-tuning with LoRA
"""

from typing import Any, Dict, Optional, Sequence, Tuple

try:
    import pytorch_lightning as pl
    from pytorch_lightning.trainer.trainer import Trainer
except ImportError:
    pl = None  # type: ignore
    Trainer = None  # type: ignore

import torch
import torch.nn as nn
import torch.nn.functional as F


try:
    from monai.networks.nets.swin_unetr import SwinUNETR
except ImportError:
    SwinUNETR = None  # type: ignore

from augmentations.mask import random_mask


class ContrastiveTransformer(pl.LightningModule):  # type: ignore
    """
    Self-supervised learning model combining Masked Autoencoding (MAE) and
    Momentum Contrastive Learning (MoCo) for 3D medical images.

    This model learns robust visual representations by:
    1. Reconstructing masked input volumes (MAE objective)
    2. Maximizing agreement between augmented views (MoCo objective)

    Args:
        patch_size: Size of patches for Swin Transformer (default: (4, 4, 4))
        learning_rate: Initial learning rate (default: 1e-4)
        img_size: Input image dimensions (default: (96, 96, 96))
        feature_size: Base feature dimension for Swin-UNETR (default: 24)
        mask_ratio: Ratio of volume to mask for MAE (default: 0.6)
        temperature: Temperature parameter for contrastive loss (default: 0.6)
        queue_size: Size of negative sample queue for MoCo (default: 4096)
        momentum: Momentum coefficient for updating key encoder (default: 0.996)
        warmup_epochs: Number of warmup epochs (default: 1)
        max_epochs: Total number of training epochs (default: 30)
        min_lr: Minimum learning rate for cosine annealing (default: 1e-5)
    """

    def __init__(
        self,
        patch_size: Sequence[int] = (4, 4, 4),
        learning_rate: float = 1e-4,
        img_size: Tuple[int, int, int] = (96, 96, 96),
        feature_size: int = 24,
        mask_ratio: float = 0.6,
        temperature: float = 0.6,
        queue_size: int = 4096,
        momentum: float = 0.996,
        warmup_epochs: int = 1,
        max_epochs: int = 30,
        min_lr: float = 1e-5,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        # Store hyperparameters as instance attributes
        self.warmup_epochs: int = warmup_epochs
        self.max_epochs: int = max_epochs
        self.min_lr: float = min_lr
        self.temperature: float = temperature
        self.momentum: float = momentum
        self.queue_size: int = queue_size
        self.mask_ratio: float = mask_ratio
        self.learning_rate: float = learning_rate

        # Type hints for PyTorch Lightning attributes
        self.trainer: Optional[Any]  # Trainer type from pl

        if SwinUNETR is None:
            raise ImportError("monai package is required for SwinUNETR")

        # Query encoder
        self.encoder: Any = SwinUNETR(
            in_channels=1,
            out_channels=1,
            feature_size=feature_size,
            use_checkpoint=True,
            use_v2=True,
        )

        # Key encoder (momentum-updated)
        self.encoder_m: Any = SwinUNETR(
            in_channels=1,
            out_channels=1,
            feature_size=feature_size,
            use_checkpoint=True,
            use_v2=True,
        )

        # Initialize momentum encoder with query encoder weights
        for param_q, param_k in zip(
            self.encoder.parameters(), self.encoder_m.parameters()
        ):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False

        # Determine encoder output dimension
        with torch.no_grad():
            dummy_input: torch.Tensor = torch.zeros(1, 1, *img_size)
            features: torch.Tensor = self.encoder.swinViT(dummy_input)[-1]
            encoder_dim: int = features.shape[1]

        # Query projection head
        self.projection: nn.Sequential = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Flatten(),
            nn.Linear(encoder_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
        )

        # Key projection head
        self.projection_m: nn.Sequential = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Flatten(),
            nn.Linear(encoder_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
        )

        # Initialize momentum projection with query projection weights
        for param_q, param_k in zip(
            self.projection.parameters(), self.projection_m.parameters()
        ):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False

        # MoCo queue for storing negative samples
        self.register_buffer("queue", F.normalize(torch.randn(128, queue_size), dim=0))
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

        # Type hints for buffers (helps Pylance)
        self.queue: torch.Tensor
        self.queue_ptr: torch.Tensor

    @torch.no_grad()
    def _momentum_update(self) -> None:
        """
        Update momentum encoder and projection head using exponential moving
        average.
        """
        for param_q, param_k in zip(
            self.encoder.parameters(), self.encoder_m.parameters()
        ):
            param_k.data = param_k.data * self.momentum + param_q.data * (
                1.0 - self.momentum
            )
        for param_q, param_k in zip(
            self.projection.parameters(), self.projection_m.parameters()
        ):
            param_k.data = param_k.data * self.momentum + param_q.data * (
                1.0 - self.momentum
            )

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys: torch.Tensor) -> None:
        """
        Update the MoCo queue with new key features.

        Args:
            keys: New key features to add to the queue
        """
        if self.trainer is not None and self.trainer.world_size > 1:
            keys = self._concat_all_gather(keys)

        batch_size: int = keys.shape[0]
        ptr: int = int(self.queue_ptr)

        if ptr + batch_size > self.queue_size:
            self.queue[:, ptr:] = keys[: self.queue_size - ptr].T
            self.queue[:, : batch_size - (self.queue_size - ptr)] = keys[
                self.queue_size - ptr :
            ].T
            ptr = batch_size - (self.queue_size - ptr)
        else:
            self.queue[:, ptr : ptr + batch_size] = keys.T
            ptr = (ptr + batch_size) % self.queue_size

        self.queue_ptr[0] = ptr

    @torch.no_grad()
    def _concat_all_gather(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Gather tensors from all distributed processes.

        Args:
            tensor: Tensor to gather across processes

        Returns:
            Concatenated tensor from all processes
        """
        # Check if distributed is available and initialized
        import torch.distributed as dist  # type: ignore
        from typing import List

        if not dist.is_available():
            return tensor

        if not dist.is_initialized():
            return tensor

        tensors_gather: List[torch.Tensor] = [
            torch.zeros_like(tensor) for _ in range(dist.get_world_size())
        ]
        dist.all_gather(tensors_gather, tensor, async_op=False)
        return torch.cat(tensors_gather, dim=0)

    def forward_encoder(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through encoder only.

        Args:
            x: Input tensor

        Returns:
            Encoded features
        """
        features: torch.Tensor = self.encoder.swinViT(x)[-1]
        return features

    def forward_mae(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Masked autoencoding forward pass.

        Args:
            x: Input tensor

        Returns:
            Tuple of (reconstruction, mask)
        """
        masked_x: torch.Tensor
        mask: torch.Tensor
        masked_x, mask = random_mask(x, self.mask_ratio, 4)
        reconstruction: torch.Tensor = self.encoder(masked_x)
        return reconstruction, mask

    def forward_contrastive(self, x: torch.Tensor) -> torch.Tensor:
        """
        Contrastive learning forward pass through query encoder.

        Args:
            x: Input tensor

        Returns:
            Normalized query embeddings
        """
        features: torch.Tensor = self.forward_encoder(x)
        z: torch.Tensor = self.projection(features)
        return F.normalize(z, dim=1)

    @torch.no_grad()
    def forward_momentum(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through momentum encoder.

        Args:
            x: Input tensor

        Returns:
            Normalized key embeddings
        """
        features: torch.Tensor = self.encoder_m.swinViT(x)[-1]
        z: torch.Tensor = self.projection_m(features)
        return F.normalize(z, dim=1)

    def contrastive_loss(self, q: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
        """
        Compute InfoNCE loss for MoCo with numerical stability.

        Args:
            q: Query embeddings
            k: Key embeddings

        Returns:
            Contrastive loss value
        """
        # Compute positive logits
        l_pos: torch.Tensor = torch.einsum("nc,nc->n", [q, k]).unsqueeze(-1)
        # Compute negative logits
        l_neg: torch.Tensor = torch.einsum(
            "nc,ck->nk", [q, self.queue.clone().detach()]
        )

        # Clamp for numerical stability
        l_pos = torch.clamp(l_pos, min=-1.0, max=1.0)
        l_neg = torch.clamp(l_neg, min=-1.0, max=1.0)

        # Concatenate and scale by temperature
        logits: torch.Tensor = torch.cat([l_pos, l_neg], dim=1) / self.temperature

        # Check for numerical issues
        if torch.isnan(logits).any() or torch.isinf(logits).any():
            self.log("debug/nan_detected", 1.0)
            self.log("debug/l_pos_min", l_pos.min())
            self.log("debug/l_pos_max", l_pos.max())
            self.log("debug/l_neg_min", l_neg.min())
            self.log("debug/l_neg_max", l_neg.max())

        labels: torch.Tensor = torch.zeros(
            logits.size(0), dtype=torch.long, device=logits.device
        )
        return F.cross_entropy(logits, labels)

    def training_step(
        self, batch: Dict[str, torch.Tensor], batch_idx: int, dataloader_idx: int = 0
    ) -> torch.Tensor:
        """
        Training step combining MAE and contrastive learning.

        Args:
            batch: Dictionary containing either:
                   - 'vol1' and 'vol2' (contrastive pairs) when dataloader_idx=0
                   - 'volume' (single volume) when dataloader_idx=1 (MAE only)
            batch_idx: Batch index
            dataloader_idx: Index of the dataloader (0=contrastive+MAE, 1=MAE only)

        Returns:
            Total loss
        """
        # Dataloader 0: Contrastive pairs (vol1, vol2) - apply both MAE and contrastive loss
        if dataloader_idx == 0:
            view1: torch.Tensor = batch["vol1"]
            view2: torch.Tensor = batch["vol2"]

            # MAE loss on both views
            recon1: torch.Tensor
            mask1: torch.Tensor
            recon1, mask1 = self.forward_mae(view1)
            recon2: torch.Tensor
            mask2: torch.Tensor
            recon2, mask2 = self.forward_mae(view2)
            loss_view1: torch.Tensor = F.mse_loss(recon1[mask1], view1[mask1])
            loss_view2: torch.Tensor = F.mse_loss(recon2[mask2], view2[mask2])
            mae_loss: torch.Tensor = 0.5 * (loss_view1 + loss_view2)

            # Update momentum encoder
            self._momentum_update()

            # Compute query embeddings
            q1: torch.Tensor = self.forward_contrastive(view1)
            q2: torch.Tensor = self.forward_contrastive(view2)

            # Compute key embeddings (no gradient)
            with torch.no_grad():
                k1: torch.Tensor = self.forward_momentum(view1)
                k2: torch.Tensor = self.forward_momentum(view2)

            # Cross-view contrastive loss
            loss_12: torch.Tensor = self.contrastive_loss(q1, k2)
            loss_21: torch.Tensor = self.contrastive_loss(q2, k1)
            contrastive_loss: torch.Tensor = 0.5 * (loss_12 + loss_21)

            # Update queue
            self._dequeue_and_enqueue(torch.cat([k1, k2]))

            # Total loss
            total_loss: torch.Tensor = mae_loss + contrastive_loss

            # Logging
            self.log_dict(
                {
                    "train/loss_contrastive": total_loss,
                    "train/mae_loss_contrastive": mae_loss,
                    "train/contrastive_loss": contrastive_loss,
                },
                prog_bar=True,
                on_step=True,
                on_epoch=True,
                sync_dist=True,
            )

            return total_loss

        # Dataloader 1: Single volumes (MAE only, includes scan_* files)
        else:
            volume: torch.Tensor = batch["volume"]

            # MAE loss only
            recon: torch.Tensor
            mask: torch.Tensor
            recon, mask = self.forward_mae(volume)
            mae_loss: torch.Tensor = F.mse_loss(recon[mask], volume[mask])

            # Logging
            self.log_dict(
                {
                    "train/loss_mae_only": mae_loss,
                    "train/mae_loss_only": mae_loss,
                },
                prog_bar=True,
                on_step=True,
                on_epoch=True,
                sync_dist=True,
            )

            return mae_loss

    def validation_step(
        self, batch: Dict[str, torch.Tensor], batch_idx: int, dataloader_idx: int = 0
    ) -> torch.Tensor:
        """
        Validation step.

        Args:
            batch: Dictionary containing either:
                   - 'vol1' and 'vol2' (contrastive pairs) when dataloader_idx=0
                   - 'volume' (single volume) when dataloader_idx=1 (MAE only)
            batch_idx: Batch index
            dataloader_idx: Index of the dataloader (0=contrastive+MAE, 1=MAE only)

        Returns:
            Total loss
        """
        # Dataloader 0: Contrastive pairs (vol1, vol2) - apply both MAE and contrastive loss
        if dataloader_idx == 0:
            view1: torch.Tensor = batch["vol1"]
            view2: torch.Tensor = batch["vol2"]

            # MAE loss
            recon1: torch.Tensor
            mask1: torch.Tensor
            recon1, mask1 = self.forward_mae(view1)
            recon2: torch.Tensor
            mask2: torch.Tensor
            recon2, mask2 = self.forward_mae(view2)
            loss_view1: torch.Tensor = F.mse_loss(recon1[mask1], view1[mask1])
            loss_view2: torch.Tensor = F.mse_loss(recon2[mask2], view2[mask2])
            mae_loss: torch.Tensor = 0.5 * (loss_view1 + loss_view2)

            # Contrastive loss (no momentum update in validation)
            q1: torch.Tensor = self.forward_contrastive(view1)
            q2: torch.Tensor = self.forward_contrastive(view2)
            k1: torch.Tensor = self.forward_momentum(view1)
            k2: torch.Tensor = self.forward_momentum(view2)

            loss_12: torch.Tensor = self.contrastive_loss(q1, k2)
            loss_21: torch.Tensor = self.contrastive_loss(q2, k1)
            contrastive_loss: torch.Tensor = 0.5 * (loss_12 + loss_21)

            total_loss: torch.Tensor = mae_loss + contrastive_loss

            self.log_dict(
                {
                    "val/loss_contrastive": total_loss,
                    "val/mae_loss_contrastive": mae_loss,
                    "val/contrastive_loss": contrastive_loss,
                },
                prog_bar=False,
                on_epoch=True,
                sync_dist=True,
            )

            return total_loss

        # Dataloader 1: Single volumes (MAE only, includes scan_* files)
        else:
            volume: torch.Tensor = batch["volume"]

            # MAE loss only
            recon: torch.Tensor
            mask: torch.Tensor
            recon, mask = self.forward_mae(volume)
            mae_loss: torch.Tensor = F.mse_loss(recon[mask], volume[mask])

            self.log_dict(
                {
                    "val/loss_mae_only": mae_loss,
                    "val/mae_loss_only": mae_loss,
                },
                prog_bar=False,
                on_epoch=True,
                sync_dist=True,
            )

            return mae_loss

    def configure_optimizers(self) -> Any:
        """
        Configure optimizer and learning rate scheduler.

        Returns:
            Dictionary containing optimizer and scheduler configuration
        """
        optimizer: torch.optim.AdamW = torch.optim.AdamW(
            self.parameters(), lr=self.learning_rate, weight_decay=0.01
        )

        # Calculate total steps
        if self.trainer is None:
            raise RuntimeError(
                "Trainer must be set before calling configure_optimizers"
            )
        num_training_steps: int = self.trainer.estimated_stepping_batches
        num_warmup_steps: int = int(
            num_training_steps * self.warmup_epochs / self.max_epochs
        )

        # Warmup scheduler
        warmup_scheduler: torch.optim.lr_scheduler.LinearLR = (
            torch.optim.lr_scheduler.LinearLR(
                optimizer,
                start_factor=1e-6,
                end_factor=1.0,
                total_iters=num_warmup_steps,
            )
        )

        # Cosine decay scheduler
        cosine_scheduler: torch.optim.lr_scheduler.CosineAnnealingLR = (
            torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=(num_training_steps - num_warmup_steps),
                eta_min=self.min_lr,
            )
        )

        # Chain schedulers together
        lr_scheduler: torch.optim.lr_scheduler.SequentialLR = (
            torch.optim.lr_scheduler.SequentialLR(
                optimizer,
                schedulers=[warmup_scheduler, cosine_scheduler],
                milestones=[num_warmup_steps],
            )
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": lr_scheduler,
                "interval": "step",
            },
        }
