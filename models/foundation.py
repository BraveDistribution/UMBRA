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

from __future__ import annotations

__all__ = [
    "MAEPretrainer",
    "ContrastiveMAEPretrainer",
]

from typing import Any, Callable, Dict, Optional, Sequence, Tuple, Union, List, Literal
from typing import cast
from copy import deepcopy

import lightning.pytorch as pl
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.networks import SwinMAE
from utils.masking import generate_random_mask_conv, up_to_voxel_space
from utils.misc import ensure_tuple_dim, schedule_param
from utils.metrics import effective_rank


class MAEPretrainer(pl.LightningModule):  # type: ignore
    """
    Base class for Masked Autoencoder (MAE) pretraining on 3D medical images.

    This model learns visual representations by reconstructing masked input volumes.
    The encoder is based on MONAI's `SwinTransformer` architecture.
    """
    def __init__(
        self,
        *,
        # Encoder args
        in_channels: int = 1,
        patch_size: Union[int, Sequence[int]] = 2,
        depths: Sequence[int] = (2, 2, 6, 2),
        num_heads: Sequence[int] = (3, 6, 12, 24),
        window_size: Union[Sequence[int], int] = 7,
        feature_size: int = 48,
        use_v2: bool = True,
        spatial_dims: int = 3,
        use_checkpoint: bool = True,
        extra_swin_kwargs: Optional[Dict[str, Any]] = None,
        gn_groups: int = 8,
        # Decoder args
        width: int = 32,
        # Weight init
        weight_init_fn: Optional[Callable[[nn.Module], None]] = None,
        # Masking args
        mask_ratio: Union[float, Sequence[float]] = [0.6, 0.75],
        input_size: Union[int, Sequence[int]] = 96,
        # Optimizer args
        learning_rate: float = 1e-4,
        min_lr: float = 1e-5,
        warmup: Union[int, float] = 0.02,
    ) -> None:
        """
        Args:
            in_channels:        Number of input channels
            patch_size:         Patch size for `SwinTransformer`
            depths:             Depth of the `SwinTransformer`
            num_heads:          Number of attention heads for `SwinTransformer`
            window_size:        Size of the window for `SwinTransformer`
            feature_size:       Feature size for `SwinTransformer`
            use_v2:             Use v2 version of `SwinTransformer`
            spatial_dims:       Spatial dimensions
            use_checkpoint:     Use checkpointing for `SwinTransformer`
            extra_swin_kwargs:  Extra keyword arguments for `SwinTransformer`
            gn_groups:          Number of groups for GroupNorm
            width:              Width of the FPN decoder
            weight_init_fn:     Function to initialize weights
            mask_ratio:         Ratio of volume to mask for MAE
            input_size:         Model input size
            learning_rate:      Initial learning rate
            min_lr:             Minimum learning rate for cosine annealing
            warmup:             Number of warmup steps if int, or fraction of total steps if float
        """
        super().__init__()
        self.save_hyperparameters()

        # Store hyperparameters as instance attributes
        self.warmup: Union[int, float] = warmup
        self.min_lr: float = min_lr
        self.learning_rate: float = learning_rate
        self.mask_ratio: Sequence[float] = ensure_tuple_dim(mask_ratio, 2)
        self.patch_size: Sequence[int] = ensure_tuple_dim(patch_size, spatial_dims)
        self.num_downsamples: int = len(depths)
        self.input_size: Sequence[int] = ensure_tuple_dim(input_size, spatial_dims)

        # Type hints for PyTorch Lightning attributes
        self.trainer: Optional[Any]

        # Encoder
        self.model: Any = SwinMAE(
            in_channels=in_channels,
            patch_size=patch_size,
            depths=depths,
            num_heads=num_heads,
            window_size=window_size,
            feature_size=feature_size,
            use_v2=use_v2,
            spatial_dims=spatial_dims,
            use_checkpoint=use_checkpoint,
            extra_swin_kwargs=extra_swin_kwargs,
            gn_groups=gn_groups,
            width=width,
        )

        # Initialize weights
        if weight_init_fn is not None:
            try:
                self.model.apply(weight_init_fn)
            except Exception:
                print("Weight initialization function failed for encoder; skipping.")

    @property
    def encoder(self) -> nn.Module:
        return self.model.encoder

    def forward_mae(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Masked autoencoding forward pass.

        Args:
            x: Input tensor

        Returns:
            Tuple of (recon, mask)
        """
        # Generate mask at patch level
        mask_patch: torch.Tensor = generate_random_mask_conv(
            input_size=self.input_size,
            patch_size=self.patch_size,
            num_downsamples=self.num_downsamples,
            batch_size=x.shape[0],
            mask_ratio=self.mask_ratio,
            return_kind="grid",
            device=x.device,
        )
        pred: torch.Tensor = self.model(x, mask_patch)

        # Upsample mask to voxel level for reconstruction loss
        mask_voxel: torch.Tensor = up_to_voxel_space(
            mask_finest_grid=mask_patch,
            original_spatial=self.input_size,
            patch_size=self.patch_size,
        )

        return pred, mask_voxel

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
        target: torch.Tensor = batch["volume_recon"]

        # MAE loss
        recon: torch.Tensor
        mask: torch.Tensor
        recon, mask = self.forward_mae(volume)
        mae_loss: torch.Tensor = F.mse_loss(recon[mask], target[mask])

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
        target: torch.Tensor = batch["volume_recon"]

        # MAE loss
        recon: torch.Tensor
        mask: torch.Tensor
        recon, mask = self.forward_mae(volume)
        mae_loss: torch.Tensor = F.mse_loss(recon[mask], target[mask])

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
        num_warmup_steps: int = (
            int(num_training_steps * self.warmup) if isinstance(self.warmup, float) 
            else self.warmup
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
    """
    def __init__(
        self,
        *,
        # Encoder args
        in_channels: int = 1,
        patch_size: Union[int, Sequence[int]] = 2,
        depths: Sequence[int] = (2, 2, 6, 2),
        num_heads: Sequence[int] = (3, 6, 12, 24),
        window_size: Union[Sequence[int], int] = 7,
        feature_size: int = 48,
        use_v2: bool = True,
        spatial_dims: int = 3,
        use_checkpoint: bool = True,
        extra_swin_kwargs: Optional[Dict[str, Any]] = None,
        weight_init_fn: Optional[Callable[[nn.Module], None]] = None,
        # Masking args
        mask_ratio: Union[float, Sequence[float]] = [0.6, 0.75],
        input_size: Union[int, Sequence[int]] = 96,
        # Contrastive learning args
        temperature: float = 0.2,
        queue_size: int = 16384,
        momentum_kwargs: Union[Dict[str, Any], float] = {
            "start_frac": 0.05,
            "end_frac": 0.2,
            "start_val": 0.996,
            "end_val": 0.999,
            "mode": "linear",
        },
        # Combined mode args
        combined_kwargs: Union[Dict[str, Any], float] = {
            "start_frac": 0.05,
            "end_frac": 0.1,
            "start_val": 0.1,
            "end_val": 1.0,
            "mode": "linear",
        },
        # Optimizer args
        learning_rate: float = 1e-4,
        min_lr: float = 1e-5,
        warmup: Union[int, float] = 0.02,
        # Pretraining mode
        pretraining_mode: Literal["contrastive_only", "combined"] = "contrastive_only",
    ) -> None:
        """
        Args:
            in_channels:        Number of input channels
            patch_size:         Patch size for `SwinTransformer`
            depths:             Depth of the SwinTransformer
            num_heads:          Number of attention heads for SwinTransformer
            window_size:        Size of the window for SwinTransformer
            feature_size:       Feature size for SwinTransformer
            use_v2:             Use v2 version of SwinTransformer
            spatial_dims:       Spatial dimensions
            use_checkpoint:     Use checkpointing for SwinTransformer
            extra_swin_kwargs:  Extra keyword arguments for SwinTransformer
            weight_init_fn:     Function to initialize weights
            mask_ratio:         Ratio of volume to mask for MAE
            input_size:         Model input size
            temperature:        Temperature parameter for contrastive loss
            queue_size:         Size of negative sample queue for MoCo
            momentum_kwargs:    Keyword arguments for scheduling momentum coefficient using the 
                                    `utils.misc.schedule_param` function. Can replace with a constant value.
            combined_kwargs:    Keyword arguments for scheduling the combined loss weights using the
                                    `utils.misc.schedule_param` function, so that the combined loss is: 
                                    `loss = alpha * contrastive_loss + mae_loss`.
                                    Can replace with a constant value.
            learning_rate:      Initial learning rate
            min_lr:             Minimum learning rate for cosine annealing
            warmup:             Number of warmup steps if int, or fraction of total steps if float
            pretraining_mode:   Pretraining mode
        """
        # Early failure
        if pretraining_mode not in ["contrastive_only", "combined"]:
            raise ValueError(
                f"Invalid pretraining mode: {pretraining_mode}. "
                "Must be one of: 'contrastive_only', 'combined'"
            )

        # Initialize base class (MAE components)
        super().__init__(
            in_channels=in_channels,
            patch_size=patch_size,
            depths=depths,
            num_heads=num_heads,
            window_size=window_size,
            feature_size=feature_size,
            use_v2=use_v2,
            spatial_dims=spatial_dims,
            use_checkpoint=use_checkpoint,
            extra_swin_kwargs=extra_swin_kwargs,
            weight_init_fn=weight_init_fn,
            input_size=input_size,
            mask_ratio=mask_ratio,
            learning_rate=learning_rate,
            min_lr=min_lr,
            warmup=warmup,
        )

        # Store extra (contrastive + pretraining mode) hyperparameters
        self.temperature: float = temperature
        self.momentum: Union[Dict[str, Any], float] = momentum_kwargs
        self.queue_size: int = queue_size
        self.pretraining_mode: str = pretraining_mode
        self.combined_kwargs: Union[Dict[str, Any], float] = combined_kwargs

        # Initialize momentum (key) encoder with query encoder weights
        self.encoder_m = deepcopy(self.encoder)
        for param in self.encoder_m.parameters():
            param.requires_grad = False

        # Deterministic encoder output dimension for Swin Transformer
        encoder_dim: int = feature_size * 2**(len(depths))

        # Query projection head + optional weight init
        self.projection: nn.Sequential = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Flatten(),
            nn.Linear(encoder_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
        )
        if weight_init_fn is not None:
            try:
                self.projection.apply(weight_init_fn)
            except Exception:
                print("Weight initialization function failed for projection; skipping.")

        # Initialize momentum (key) projection head with query projection weights
        self.projection_m: nn.Sequential = deepcopy(self.projection)
        for param in self.projection_m.parameters():
            param.requires_grad = False

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
        if self.trainer is None:
            raise RuntimeError(
                "Trainer must be set before calling configure_optimizers"
            )

        m: float = (
            schedule_param(
                current=self.trainer.global_step,
                max_steps=self.trainer.estimated_stepping_batches,
                **self.momentum,
            ) if isinstance(self.momentum, dict) else self.momentum
        )

        for param_q, param_k in zip(
            self.encoder.parameters(), self.encoder_m.parameters()
        ):
            param_k.data = param_k.data * m + param_q.data * (1.0 - m)
            
        for param_q, param_k in zip(
            self.projection.parameters(), self.projection_m.parameters()
        ):
            param_k.data = param_k.data * m + param_q.data * (1.0 - m)

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

        if not dist.is_available():
            return tensor

        if not dist.is_initialized():
            return tensor

        tensors_gather: List[torch.Tensor] = [
            torch.zeros_like(tensor) for _ in range(dist.get_world_size())
        ]
        dist.all_gather(tensors_gather, tensor, async_op=False)
        return torch.cat(tensors_gather, dim=0)

    def forward_contrastive(self, x: torch.Tensor) -> torch.Tensor:
        """
        Contrastive learning forward pass through query encoder.

        Args:
            x: Input tensor

        Returns:
            Normalized query embeddings
        """
        features: torch.Tensor = self.encoder(x)[-1]
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
        features: torch.Tensor = self.encoder_m(x)[-1]
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

        # Create valid queue mask: True where queue entry has been initialized (patient_id != 0)
        valid_queue_mask = queue_patient_ids != 0  # (queue_size,)

        # Create patient-mismatch mask: True where queue patient != current patient
        patient_mismatch_mask = patient_ids.unsqueeze(1) != queue_patient_ids.unsqueeze(0) # (B, queue_size)

        # Combine masks to get valid negatives
        valid_negative_mask = patient_mismatch_mask & valid_queue_mask.unsqueeze(0) # (B, queue_size)
        num_valid_negatives = valid_negative_mask.sum(dim=1) # (B,); for debugging/monitoring

        # Apply mask: set invalid negatives to very negative value
        l_neg = torch.where(valid_negative_mask, l_neg, torch.tensor(-10.0, device=l_neg.device))

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

        # Log number of valid negatives per query for debugging/monitoring
        self.log("debug/mean_negatives", num_valid_negatives.float().mean(),
            on_step=True, on_epoch=False, rank_zero_only=True, sync_dist=False)

        labels: torch.Tensor = torch.zeros(
            logits.size(0), dtype=torch.long, device=logits.device
        )
        return F.cross_entropy(logits, labels)

    def _process_contrastive_batch(
        self, batch: Dict[str, torch.Tensor], is_training: bool = True
    ) -> torch.Tensor:
        """Process a contrastive batch and return the loss."""
        if self.trainer is None:
            raise RuntimeError(
                "Trainer must be set before calling _process_contrastive_batch"
            )

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
            target1: torch.Tensor = batch["vol1_recon"]
            target2: torch.Tensor = batch["vol2_recon"]

            recon1, mask1 = self.forward_mae(view1)
            recon2, mask2 = self.forward_mae(view2)
            loss_view1 = F.mse_loss(recon1[mask1], target1[mask1])
            loss_view2 = F.mse_loss(recon2[mask2], target2[mask2])
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
        alpha: float = (
            schedule_param(
                current=self.trainer.global_step,
                max_steps=self.trainer.estimated_stepping_batches,
                **self.combined_kwargs,
            ) if isinstance(self.combined_kwargs, dict) else self.combined_kwargs
        )
        if self.pretraining_mode == "contrastive_only":
            total_loss = contrastive_loss
        else:
            total_loss = alpha * contrastive_loss + mae_loss

        # Logging
        prefix = "train" if is_training else "val"
        log_dict: Dict[str, Union[float, int, torch.Tensor]] = {
            f"{prefix}/contrastive_loss": contrastive_loss,
            f"{prefix}/effective_rank": effective_rank(self.queue, method="auto")[0],
        }
        if self.pretraining_mode == "combined":
            log_dict[f"{prefix}/mae_loss"] = mae_loss
            log_dict[f"{prefix}/combined_loss"] = total_loss
            log_dict[f"{prefix}/loss_weight"] = alpha
        self.log_dict(
            log_dict, 
            prog_bar=True, 
            on_step=True, 
            on_epoch=True, 
            sync_dist=True,
            batch_size=len(batch["vol1"]),
        )

        return total_loss

    def _process_mae_batch(
        self, batch: Dict[str, torch.Tensor], is_training: bool = True
    ) -> torch.Tensor:
        """Process an MAE batch and return the loss."""
        volume: torch.Tensor = batch["volume"]
        target: torch.Tensor = batch["volume_recon"]

        # MAE loss only
        recon, mask = self.forward_mae(volume)
        mae_loss = F.mse_loss(recon[mask], target[mask])

        # Logging
        prefix = "train" if is_training else "val"
        self.log_dict(
            {
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
            return self._process_contrastive_batch(
                cast(Dict[str, torch.Tensor], actual_batch), is_training=True
            )

        # Dataloader 1: Single volumes (MAE only, includes scan_* files)
        else:
            return self._process_mae_batch(
                cast(Dict[str, torch.Tensor], actual_batch), is_training=True
            )

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
