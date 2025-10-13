"""
Foundation models for medical image analysis.

This module contains implementations of self-supervised pre-training models
and fine-tuning architectures for 3D medical image segmentation, including:

Architecture:
- MAEPretrainer: Base class for Masked Autoencoder (MAE) pretraining
- ContrastiveMAEPretrainer: Extends MAE with MoCo-based contrastive learning
- ContrastiveTransformer: Alias for ContrastiveMAEPretrainer (backward compatibility)
- SegmentationFineTuner: Parameter-efficient fine-tuning with LoRA
"""

from typing import Any, Dict, Optional, Sequence, Tuple, Union, List

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.networks.nets.swin_unetr import SwinUNETR
from augmentations.mask import random_mask


class MAEPretrainer(pl.LightningModule):  # type: ignore
    """
    Base class for Masked Autoencoder (MAE) pretraining on 3D medical images.

    This model learns visual representations by reconstructing masked input volumes.
    The encoder is based on Swin-UNETR architecture.

    Args:
        patch_size: Size of patches for Swin Transformer (default: (4, 4, 4))
        learning_rate: Initial learning rate (default: 1e-4)
        img_size: Input image dimensions (default: (96, 96, 96))
        feature_size: Base feature dimension for Swin-UNETR (default: 24)
        mask_ratio: Ratio of volume to mask for MAE (default: 0.6)
        warmup_epochs: Number of warmup epochs (default: 1)
        max_epochs: Total number of training epochs (default: 30)
        min_lr: Minimum learning rate for cosine annealing (default: 1e-5)
        pretraining_mode: Mode for pretraining ('mae_only', 'contrastive_only', 'combined')
    """

    def __init__(
        self,
        patch_size: Sequence[int] = (4, 4, 4),
        learning_rate: float = 1e-4,
        img_size: Tuple[int, int, int] = (96, 96, 96),
        feature_size: int = 24,
        mask_ratio: float = 0.6,
        warmup_epochs: int = 1,
        max_epochs: int = 30,
        min_lr: float = 1e-5,
        pretraining_mode: str = "mae_only",
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        # Store hyperparameters as instance attributes
        self.warmup_epochs: int = warmup_epochs
        self.max_epochs: int = max_epochs
        self.min_lr: float = min_lr
        self.mask_ratio: float = mask_ratio
        self.learning_rate: float = learning_rate
        self.pretraining_mode: str = pretraining_mode

        # Type hints for PyTorch Lightning attributes
        self.trainer: Optional[Any]

        if SwinUNETR is None:
            raise ImportError("monai package is required for SwinUNETR")

        # Encoder (Swin-UNETR)
        self.encoder: Any = SwinUNETR(
            in_channels=1,
            out_channels=1,
            feature_size=feature_size,
            use_checkpoint=True,
            use_v2=True,
        )

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

    def forward_mae(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Masked autoencoding forward pass.

        Args:
            x: Input tensor

        Returns:
            Tuple of (reconstruction, mask, original)
        """
        original: torch.Tensor = x.clone()
        masked_x: torch.Tensor
        mask: torch.Tensor
        masked_x, mask = random_mask(x, self.mask_ratio, 4)
        reconstruction: torch.Tensor = self.encoder(masked_x)
        return reconstruction, mask, original

    def training_step(
        self, batch: Dict[str, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """
        Training step for MAE.

        Args:
            batch: Dictionary containing 'volume' key
            batch_idx: Batch index

        Returns:
            MAE loss
        """
        volume: torch.Tensor = batch["volume"]

        # MAE loss
        recon: torch.Tensor
        mask: torch.Tensor
        original: torch.Tensor
        recon, mask, original = self.forward_mae(volume)
        mae_loss: torch.Tensor = F.mse_loss(recon[mask], original[mask])

        # Logging
        self.log_dict(
            {
                "train/loss_mae": mae_loss,
                "train/mae_loss": mae_loss,
            },
            prog_bar=True,
            on_step=True,
            on_epoch=True,
            sync_dist=True,
        )

        return mae_loss

    def validation_step(
        self, batch: Dict[str, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """
        Validation step for MAE.

        Args:
            batch: Dictionary containing 'volume' key
            batch_idx: Batch index

        Returns:
            MAE loss
        """
        volume: torch.Tensor = batch["volume"]

        # MAE loss
        recon: torch.Tensor
        mask: torch.Tensor
        original: torch.Tensor
        recon, mask, original = self.forward_mae(volume)
        mae_loss: torch.Tensor = F.mse_loss(recon[mask], original[mask])

        # Logging
        self.log_dict(
            {
                "val/loss_mae": mae_loss,
                "val/mae_loss": mae_loss,
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


class ContrastiveMAEPretrainer(MAEPretrainer):  # type: ignore
    """
    Extends MAEPretrainer with Momentum Contrastive Learning (MoCo).

    This model learns robust visual representations by:
    1. Reconstructing masked input volumes (MAE objective from base class)
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
        pretraining_mode: Mode for pretraining ('mae_only', 'contrastive_only', 'combined')
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
        pretraining_mode: str = "contrastive_only",
    ) -> None:
        # Initialize base class (MAE components)
        super().__init__(
            patch_size=patch_size,
            learning_rate=learning_rate,
            img_size=img_size,
            feature_size=feature_size,
            mask_ratio=mask_ratio,
            warmup_epochs=warmup_epochs,
            max_epochs=max_epochs,
            min_lr=min_lr,
            pretraining_mode=pretraining_mode,
        )

        # Store contrastive-specific hyperparameters
        self.temperature: float = temperature
        self.momentum: float = momentum
        self.queue_size: int = queue_size

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

        # Store patient/session metadata for queue (for filtering cross-session negatives)
        # Shape: (queue_size, 2) where columns are [patient_id, session_id]
        self.register_buffer(
            "queue_metadata", torch.zeros(queue_size, 2, dtype=torch.long)
        )

        # Type hints for buffers (helps Pylance)
        self.queue: torch.Tensor
        self.queue_ptr: torch.Tensor
        self.queue_metadata: torch.Tensor

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
    def _dequeue_and_enqueue(self, keys: torch.Tensor, metadata: torch.Tensor) -> None:
        """
        Update the MoCo queue with new key features and their metadata.

        Args:
            keys: New key features to add to the queue
            metadata: Patient/session metadata (batch_size, 2) where columns are [patient_id, session_id]
        """
        if self.trainer is not None and self.trainer.world_size > 1:
            keys = self._concat_all_gather(keys)
            metadata = self._concat_all_gather(metadata)

        batch_size: int = keys.shape[0]
        ptr: int = int(self.queue_ptr)

        if ptr + batch_size > self.queue_size:
            self.queue[:, ptr:] = keys[: self.queue_size - ptr].T
            self.queue[:, : batch_size - (self.queue_size - ptr)] = keys[
                self.queue_size - ptr :
            ].T
            self.queue_metadata[ptr:, :] = metadata[: self.queue_size - ptr]
            self.queue_metadata[: batch_size - (self.queue_size - ptr), :] = metadata[
                self.queue_size - ptr :
            ]
            ptr = batch_size - (self.queue_size - ptr)
        else:
            self.queue[:, ptr : ptr + batch_size] = keys.T
            self.queue_metadata[ptr : ptr + batch_size, :] = metadata
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

    def forward_mae(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Masked autoencoding forward pass.

        Args:
            x: Input tensor

        Returns:
            Tuple of (reconstruction, mask, original)
        """
        original: torch.Tensor = x.clone()
        masked_x: torch.Tensor
        mask: torch.Tensor
        masked_x, mask = random_mask(x, self.mask_ratio, 4)
        reconstruction: torch.Tensor = self.encoder(masked_x)
        return reconstruction, mask, original

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

    def contrastive_loss(
        self, q: torch.Tensor, k: torch.Tensor, patient_ids: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute InfoNCE loss for MoCo with numerical stability and cross-session filtering.

        Args:
            q: Query embeddings (batch_size, embedding_dim)
            k: Key embeddings (batch_size, embedding_dim)
            patient_ids: Patient IDs for current batch (batch_size,)

        Returns:
            Contrastive loss value
        """
        # Compute positive logits
        l_pos: torch.Tensor = torch.einsum("nc,nc->n", [q, k]).unsqueeze(-1)
        # Compute negative logits
        l_neg: torch.Tensor = torch.einsum(
            "nc,ck->nk", [q, self.queue.clone().detach()]
        )

        # Filter out negatives from the same patient
        # queue_metadata[:, 0] contains patient IDs
        queue_patient_ids = self.queue_metadata[:, 0]  # (queue_size,)

        # Create mask: True where queue patient != current patient
        # Shape: (batch_size, queue_size)
        mask = patient_ids.unsqueeze(1) != queue_patient_ids.unsqueeze(0)

        # Apply mask: set same-patient negatives to very negative value
        l_neg = torch.where(mask, l_neg, torch.tensor(-10.0, device=l_neg.device))

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

    def _process_contrastive_batch(
        self, batch: Dict[str, torch.Tensor], is_training: bool = True
    ) -> torch.Tensor:
        """Process a contrastive batch and return the loss."""
        view1: torch.Tensor = batch["vol1"]
        view2: torch.Tensor = batch["vol2"]

        # Extract patient and session metadata
        patient_ids: torch.Tensor = batch["patient"]
        session_ids: torch.Tensor = batch["session"]

        # Convert string IDs to integers for storage
        patient_ids_int = torch.tensor(
            [int(p) for p in patient_ids], dtype=torch.long, device=view1.device
        )
        session_ids_int = torch.tensor(
            [int(s) for s in session_ids], dtype=torch.long, device=view1.device
        )

        # MAE loss on both views (only in combined mode)
        if self.pretraining_mode == "combined":
            recon1, mask1, original1 = self.forward_mae(view1)
            recon2, mask2, original2 = self.forward_mae(view2)
            loss_view1 = F.mse_loss(recon1[mask1], original1[mask1])
            loss_view2 = F.mse_loss(recon2[mask2], original2[mask2])
            mae_loss = 0.5 * (loss_view1 + loss_view2)
        else:
            mae_loss = torch.tensor(0.0, device=view1.device)

        # Update momentum encoder (only during training)
        if is_training:
            self._momentum_update()

        # Compute query embeddings
        q1 = self.forward_contrastive(view1)
        q2 = self.forward_contrastive(view2)

        # Compute key embeddings (no gradient)
        with torch.no_grad():
            k1 = self.forward_momentum(view1)
            k2 = self.forward_momentum(view2)

        # Cross-view contrastive loss with patient filtering
        loss_12 = self.contrastive_loss(q1, k2, patient_ids_int)
        loss_21 = self.contrastive_loss(q2, k1, patient_ids_int)
        contrastive_loss = 0.5 * (loss_12 + loss_21)

        # Update queue (only during training)
        if is_training:
            # Create metadata tensor for queue
            metadata = torch.stack(
                [
                    torch.cat([patient_ids_int, patient_ids_int]),
                    torch.cat([session_ids_int, session_ids_int]),
                ],
                dim=1,
            )
            self._dequeue_and_enqueue(torch.cat([k1, k2]), metadata)

        # Total loss
        if self.pretraining_mode == "contrastive_only":
            total_loss = contrastive_loss
        else:
            total_loss = mae_loss + contrastive_loss

        # Logging
        prefix = "train" if is_training else "val"
        self.log_dict(
            {
                f"{prefix}/loss_contrastive": total_loss,
                f"{prefix}/mae_loss_contrastive": mae_loss,
                f"{prefix}/contrastive_loss": contrastive_loss,
            },
            prog_bar=True,
            on_step=True,
            on_epoch=True,
            sync_dist=True,
        )

        return total_loss

    def _process_mae_batch(
        self, batch: Dict[str, torch.Tensor], is_training: bool = True
    ) -> torch.Tensor:
        """Process an MAE batch and return the loss."""
        volume: torch.Tensor = batch["volume"]

        # MAE loss only
        recon, mask, original = self.forward_mae(volume)
        mae_loss = F.mse_loss(recon[mask], original[mask])

        # Logging
        prefix = "train" if is_training else "val"
        self.log_dict(
            {
                f"{prefix}/loss_mae_only": mae_loss,
                f"{prefix}/mae_loss_only": mae_loss,
            },
            prog_bar=True,
            on_step=True,
            on_epoch=True,
            sync_dist=True,
        )

        return mae_loss

    def training_step(
        self, batch: Union[Dict[str, torch.Tensor], List[Dict[str, torch.Tensor]]], batch_idx: int, dataloader_idx: int = 0
    ) -> torch.Tensor:
        """
        Training step combining MAE and contrastive learning.

        Args:
            batch: Dictionary containing either:
                   - 'vol1' and 'vol2' (contrastive pairs) when dataloader_idx=0 (contrastive mode)
                   - 'volume' (single volume) when dataloader_idx=0 (mae_only mode) or dataloader_idx=1 (combined mode)
                   - List of [contrastive_batch, mae_batch] in combined mode
            batch_idx: Batch index
            dataloader_idx: Index of the dataloader (0=contrastive or mae_only, 1=MAE only in combined mode)

        Returns:
            Total loss
        """
        # In combined mode, PyTorch Lightning may pass batches as a list
        # We need to handle both dataloaders properly
        if self.pretraining_mode == "combined" and isinstance(batch, list):
            # Combined mode with multiple dataloaders
            # Process both dataloaders and combine losses
            contrastive_batch = batch[0]  # Contrastive pairs
            mae_batch = batch[1] if len(batch) > 1 else None  # MAE singles

            total_loss = torch.tensor(0.0, device=self.device)

            # Process contrastive batch if it has data
            if (
                contrastive_batch is not None
                and len(contrastive_batch.get("vol1", [])) > 0
            ):
                contrastive_loss = self._process_contrastive_batch(
                    contrastive_batch, is_training=True
                )
                total_loss = total_loss + contrastive_loss

            # Process MAE batch if it has data
            if mae_batch is not None and len(mae_batch.get("volume", [])) > 0:
                mae_loss = self._process_mae_batch(mae_batch, is_training=True)
                total_loss = total_loss + mae_loss

            return total_loss

        else:
            # Single dataloader mode or non-list batch
            actual_batch = batch

        # Determine batch type by checking keys
        # - Contrastive batches have 'vol1' and 'vol2'
        # - MAE batches have 'volume'
        is_mae_only_batch = (
            self.pretraining_mode == "mae_only"  # MAE-only mode
            or (
                isinstance(actual_batch, dict)
                and "volume" in actual_batch
                and "vol1" not in actual_batch
            )
        )

        # Dataloader 0 with contrastive pairs (contrastive_only or combined mode)
        if not is_mae_only_batch:
            return self._process_contrastive_batch(actual_batch, is_training=True)

        # Dataloader 1: Single volumes (MAE only, includes scan_* files)
        else:
            return self._process_mae_batch(actual_batch, is_training=True)

    def validation_step(
        self, batch: Dict[str, torch.Tensor], batch_idx: int, dataloader_idx: int = 0
    ) -> torch.Tensor:
        """
        Validation step.

        Note: In validation, PyTorch Lightning calls this method SEPARATELY for each
        dataloader with proper dataloader_idx (0, 1, 2, ...), unlike training where
        all batches are combined into a list.

        Args:
            batch: Dictionary containing either:
                   - 'vol1' and 'vol2' (contrastive pairs) when dataloader_idx=0
                   - 'volume' (single volume) when dataloader_idx=1 (combined mode)
            batch_idx: Batch index
            dataloader_idx: Index of the dataloader (0=contrastive, 1=MAE in combined mode)

        Returns:
            Total loss
        """
        # In validation, batch is always a single dict (not a list)
        # PyTorch Lightning calls validation_step separately for each dataloader
        actual_batch = batch

        # Check if we're in mae_only mode or processing MAE dataloader
        is_mae_only_batch = (
            dataloader_idx == 1  # Combined mode, MAE dataloader
            or self.pretraining_mode == "mae_only"
            or (
                isinstance(actual_batch, dict)
                and "volume" in actual_batch
                and "vol1" not in actual_batch
            )
        )

        # Dataloader 0 with contrastive pairs OR combined mode contrastive dataloader
        if not is_mae_only_batch:
            return self._process_contrastive_batch(actual_batch, is_training=False)

        # Dataloader 1: Single volumes (MAE only, includes scan_* files)
        else:
            return self._process_mae_batch(actual_batch, is_training=False)

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
