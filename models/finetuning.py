"""
Pytorch lightning modules for finetuning. 
"""

from __future__ import annotations

from typing import Union, Optional, Dict, Callable, Any, Literal, Sequence, List
from typing import cast
from pathlib import Path

import lightning.pytorch as pl
import torch.nn as nn
import torch
from monai.transforms import Compose, Activations, AsDiscrete
from monai.transforms.utility.array import Identity
from monai.inferers.inferer import SimpleInferer, SlidingWindowInferer
from monai.losses.dice import DiceCELoss
# pyright: reportPrivateImportUsage=false
from monai.metrics import (
    CumulativeIterationMetric,
    DiceMetric,
    MeanIoU,
    HausdorffDistanceMetric,
    SurfaceDistanceMetric,
)

from models.networks import SwinMAE
from models.schedulers import CosineAnnealingWithWarmup
from utils.nets import load_param_group_from_ckpt, split_decay_no_decay
from utils.misc import sync_dist_safe


class FinetuningModule(pl.LightningModule):
    """
    General purpose module for finetuning pre-trained models

    By default, the model is trained from scratch, but it can be
    configured for finetuning a pretrained encoder.
    """
    def __init__(
        self,
        *,
        model: nn.Module,
        loss_fn: nn.Module,
        metrics: Dict[str, Callable],
        learning_rate: float = 5e-4,
        min_lr: float = 5e-5,
        wd_encoder: float = 1e-3,
        wd_rest: float = 1e-2,
        warmup: Union[int, float] = 0.05,
        load_encoder_from: Optional[str] = None,
        encoder_prefix_in_ckpt: str = 'model.encoder',
        unfreeze_encoder_at: Union[int, float] = 0.0,
        encoder_lr_ratio: float = 1.0,
        input_key: str = "volume",
        target_key: str = "label",
        inferer: Optional[Callable[[torch.Tensor, nn.Module], torch.Tensor]] = None,
        postprocess: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    ):
        """
        Args:
            model: The model instance to finetune
            loss_fn: The loss function to use
            metrics: The metrics to use for performance monitoring; will sequentially
                compute the metric(s) and log the results by the provided dictionary keys.
            learning_rate: Maximum learning rate for unfrozen parameters
            min_lr: The minimum learning rate for cosine annealing
            warmup: Number of warmup steps if int, or fraction of total steps if float
            wd_encoder: Weight decay for encoder parameters
            wd_rest: Weight decay for remaining parameters
            load_encoder_from: Path to checkpoint to load encoder from
            encoder_prefix_in_ckpt: Where to find the encoder in the checkpoint (e.g., 'model.encoder')
            unfreeze_encoder_at: Step or fraction of total steps at which to unfreeze encoder.
            encoder_lr_ratio: Ratio of learning rate for the encoder; use 1.0 for equal learning rates.
            input_key: Key of the input in the batch
            target_key: Key of the target in the batch
            inferer: Callable to use for inference
        """
        super().__init__()
        self.save_hyperparameters()

        self.model: nn.Module = model
        if not hasattr(self.model, 'encoder'):
            raise ValueError(
                "Cannot infer encoder from model, which is required for finetuning; "
                "ensure model has an `encoder` attribute."
            )
        if load_encoder_from:
            self._load_encoder_from_ckpt(load_encoder_from, encoder_prefix_in_ckpt)
        self.metrics: Dict[str, Callable] = metrics
        self.loss_fn: nn.Module = loss_fn
        self.learning_rate: float = float(learning_rate)
        self.min_lr: float = float(min_lr)
        self.wd_encoder: float = float(wd_encoder)
        self.wd_rest: float = float(wd_rest)
        self.warmup: Union[int, float] = warmup
        self.unfreeze_encoder_at: Union[int, float] = unfreeze_encoder_at
        self.encoder_lr_ratio: float = float(encoder_lr_ratio)
        self.input_key: str = input_key
        self.target_key: str = target_key
        self.inferer: Callable[[torch.Tensor, nn.Module], torch.Tensor] = (
            inferer if inferer is not None else SimpleInferer()
        )
        self.postprocess: Callable[[torch.Tensor], torch.Tensor] = (
            postprocess if postprocess is not None 
            else cast(Callable[[torch.Tensor], torch.Tensor], Identity())
        )
        # Will get populated in `configure_optimizers()`
        self._unfreeze_encoder_at: Optional[int] = None

        # Type hints for PyTorch Lightning attributes
        self.trainer: Optional[Any]
    
    def _load_encoder_from_ckpt(
        self, 
        load_encoder_from: str, 
        encoder_prefix_in_ckpt: str = 'model.encoder'
    ) -> None:    
        self.model, stats = load_param_group_from_ckpt(
            self.model,
            checkpoint_path=Path(load_encoder_from),
            select_prefixes=encoder_prefix_in_ckpt,
            rename_map={encoder_prefix_in_ckpt: 'encoder'},
            strict=True,
        )
        print(f"Loaded pretrained weights from checkpoint "
                f"{load_encoder_from}\n#### Summary ####"
                f"\n- Loaded {len(stats['loaded_keys'])} parameters "
                f"(from {len(stats['loaded_keys']) + len(stats['ignored_keys'])} parameters)"
                f"\n- Unexpected {len(stats['unexpected_keys'])} parameters"
                f"\n- Missing {len(stats['missing_keys'])} parameters\n")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
        
    def on_fit_start(self) -> None:
        """Freeze encoder parameters if requested."""
        if self._unfreeze_encoder_at is None:
            raise RuntimeError(
                "Unfreeze encoder step must be set before calling `on_fit_start()`; "
                "should be populated in `configure_optimizers()`"
            )
        if self._unfreeze_encoder_at > 0:
            for p in self.model.encoder.parameters(): # type: ignore[attr-defined]
                p.requires_grad = False
            print(f"Freezing encoder parameters until step {self._unfreeze_encoder_at}.")
    
    def on_train_batch_start(self, batch, batch_idx: int) -> None:
        """Unfreeze encoder parameters at the specified step."""
        if self._unfreeze_encoder_at is None:
            raise RuntimeError(
                "Unfreeze encoder step must be set before calling `on_train_batch_start()`; "
                "should be populated in `configure_optimizers()`"
            )
        if self._unfreeze_encoder_at > 0 and self.global_step >= self._unfreeze_encoder_at:
            for p in self.model.encoder.parameters(): # type: ignore[attr-defined]
                p.requires_grad = True
            print(f"Encoder unfrozen at step {self.global_step}.")
    
    @torch.no_grad()
    def compute_metrics(self, out: torch.Tensor, target: torch.Tensor) -> dict[str, Any]:
        """Computes metrics; ignores if a metric fails."""
        stats = {}
        for name, metric in self.metrics.items():
            try:
                val = metric(out, target)
            except Exception:
                print(f"Failed to compute metric {name}; skipping.")
                continue

            # Support for MONAI's aggregated metrics
            if isinstance(metric, CumulativeIterationMetric):
                agg = metric.aggregate(reduction="mean")
                if isinstance(agg, torch.Tensor):
                    stats[name] = agg.item()
                elif isinstance(agg, list): # handle confusion matrix metrics
                    for metric_name, val in zip(metric.metric_name, agg): # type: ignore[attr-defined]
                        stats[f"{name}_{metric_name}"] = val.item()
                else:
                    raise ValueError(f"Unsupported aggregated metric type: {type(agg)}")
                metric.reset()
            else:
                stats[name] = val.mean().item()
        return stats
    
    def log_step(
        self, stage: Literal["train", "val", "test", "predict"], log_dict: dict[str, Any]
    ) -> None:
        """Logs step statistics."""
        on_step = stage == "train"
        self.log_dict({f"{stage}/{k}": v for k, v in log_dict.items()}, 
                      on_step=on_step, on_epoch=True, prog_bar=False, 
                      sync_dist=sync_dist_safe(self))

    def training_step(self, batch: dict, batch_idx: int):
        """Training step; computes loss and metrics."""
        out = self.model(batch[self.input_key])
        loss = self.loss_fn(out, batch[self.target_key])
        stats = self.compute_metrics(out, batch[self.target_key])
        stats["loss"] = loss.item()
        self.log_step("train", stats)
        return loss
    
    def validation_step(self, batch: dict, batch_idx: int):
        """Validation step; performs inference and computes metrics."""
        out = self.inferer(batch[self.input_key], self.model)
        loss = self.loss_fn(out, batch[self.target_key])
        stats = self.compute_metrics(self.postprocess(out), batch[self.target_key])
        stats["loss"] = loss.item()
        self.log_step("val", stats)
        return loss

    def test_step(self, batch: dict, batch_idx: int):
        """Test step; performs inference and computes metrics."""
        out = self.inferer(batch[self.input_key], self.model)
        stats = self.compute_metrics(self.postprocess(out), batch[self.target_key])
        self.log_step("test", stats)
        return out

    def predict_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        """Predict step; performs sliding window inference."""
        out = self.inferer(batch[self.input_key], self.model)
        return self.postprocess(out)

    def configure_optimizers(self) -> Any:
        """
        Configure optimizer and learning rate scheduler.

        Returns:
            Dictionary containing optimizer and scheduler configuration
        """
        if self.trainer is None:
            raise RuntimeError(
                "Trainer must be set before calling configure_optimizers"
            )

        # Get warmup steps
        num_training_steps: int = self.trainer.estimated_stepping_batches 
        num_warmup_steps: int = (
            int(num_training_steps * self.warmup) if isinstance(self.warmup, float) 
            else self.warmup
        )
        # Get unfreeze steps
        self._unfreeze_encoder_at = (
            int(num_training_steps * self.unfreeze_encoder_at) 
            if isinstance(self.unfreeze_encoder_at, float) 
            else self.unfreeze_encoder_at
        )
        # Unfreeze warmup steps are scaled to remaining number of steps
        unfreeze_warmup_steps: int = int(
            (num_warmup_steps / num_training_steps) * (num_training_steps - self._unfreeze_encoder_at)
        )

        # Build optimizer and scheduler
        # param groups are ordered as: encoder_decay, encoder_no_decay, rest_decay, rest_no_decay
        optimizer = torch.optim.AdamW(self._get_optimizer_param_groups())
        lr_scheduler = CosineAnnealingWithWarmup(
            optimizer,
            warmup_iters=(
                unfreeze_warmup_steps, unfreeze_warmup_steps, 
                num_warmup_steps, num_warmup_steps
            ),
            total_iters=num_training_steps,
            start_iter=(self._unfreeze_encoder_at, self._unfreeze_encoder_at, 0, 0),
            eta_min=(
                self.min_lr*self.encoder_lr_ratio, self.min_lr*self.encoder_lr_ratio, 
                self.min_lr, self.min_lr
            ),
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": lr_scheduler,
                "interval": "step",
            },
        }

    def _get_optimizer_param_groups(self) -> List[Dict[str, Any]]:
        """
        Build optimizer param groups for fine-tuning, with encoder/rest and 
        decay/no-decay splits.        
        """
        # Split model into decay / no-decay
        decay_params, no_decay_params = split_decay_no_decay(self.model)
        decay_ids = {id(p) for p in decay_params}
        nodecay_ids = {id(p) for p in no_decay_params}

        # Identify encoder parameters; regardless of requires_grad -> allows later unfreezing
        enc_param_ids = {id(p) for p in self.model.encoder.parameters()} # type: ignore[attr-defined]

        # Encoder groups
        encoder_decay_params = [
            p for p in self.model.encoder.parameters() # type: ignore[attr-defined]
            if id(p) in decay_ids
        ]
        encoder_no_decay_params = [
            p for p in self.model.encoder.parameters() # type: ignore[attr-defined]
            if id(p) in nodecay_ids
        ]

        # Remaining groups
        rest_decay_params = [
            p for p in self.model.parameters() 
            if id(p) not in enc_param_ids and id(p) in decay_ids
        ]
        rest_no_decay_params = [
            p for p in self.model.parameters() 
            if id(p) not in enc_param_ids and id(p) in nodecay_ids
        ]

        return [
            {
                "params": encoder_decay_params, "lr": self.learning_rate*self.encoder_lr_ratio, 
                "weight_decay": self.wd_encoder, "name": "encoder_decay"
            },
            {
                "params": encoder_no_decay_params, "lr": self.learning_rate*self.encoder_lr_ratio, 
                "weight_decay": 0.0, "name": "encoder_no_decay"
            },
            {
                "params": rest_decay_params, "lr": self.learning_rate, 
                "weight_decay": self.wd_rest, "name": "rest_decay"
            },
            {
                "params": rest_no_decay_params, "lr": self.learning_rate, 
                "weight_decay": 0.0, "name": "rest_no_decay"
            },
        ]
            

class SegmentationSwinFPN(FinetuningModule):
    """
    Finetuning module for segmentation tasks using Swin Transformer.
    """
    def __init__(
        self, 
        *, 
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
        width: int = 32,
        image_size: Union[int, Sequence[int]] = 96,
        learning_rate: float = 1e-4,
        min_lr: float = 1e-5,
        warmup: Union[int, float] = 0.05,
        load_encoder_from: Optional[str] = None,
        encoder_prefix_in_ckpt: str = 'model.encoder',
        unfreeze_encoder_at: Union[int, float] = 0.0,
        encoder_lr_ratio: float = 1.0,
        input_key: str = "volume",
        target_key: str = "label",
    ):  
        """
        Args:
            in_channels: Number of input channels
            patch_size: Patch size
            depths: Depth of the model
            num_heads: Number of heads
            window_size: Window size
        """
        model = SwinMAE(
            in_channels=in_channels,
            patch_size=patch_size,
            depths=depths,
            num_heads=num_heads,
            window_size=window_size,
            feature_size=feature_size,
            use_v2=use_v2,
            use_checkpoint=use_checkpoint,
            spatial_dims=spatial_dims,
            extra_swin_kwargs=extra_swin_kwargs,
            gn_groups=gn_groups,
            width=width,
        )

        loss_fn = DiceCELoss(sigmoid=True)
        metrics = {
            "dice": DiceMetric(),
            "iou": MeanIoU(),
            "hausdorff": HausdorffDistanceMetric(),
            "surface_distance": SurfaceDistanceMetric(),
        }
        inferer = cast(Callable[[torch.Tensor, nn.Module], torch.Tensor], 
                       SlidingWindowInferer(roi_size=image_size, overlap=0.25))
        postprocess = cast(
            Callable[[torch.Tensor], torch.Tensor], 
            Compose([
                Activations(sigmoid=True),
                AsDiscrete(threshold=0.5),
            ])
        )

        super().__init__(
            model=model, 
            loss_fn=loss_fn, 
            metrics=metrics, 
            learning_rate=learning_rate, 
            min_lr=min_lr, 
            warmup=warmup, 
            load_encoder_from=load_encoder_from, 
            encoder_prefix_in_ckpt=encoder_prefix_in_ckpt, 
            unfreeze_encoder_at=unfreeze_encoder_at, 
            encoder_lr_ratio=encoder_lr_ratio, 
            input_key=input_key, 
            target_key=target_key, 
            inferer=inferer, 
            postprocess=postprocess,
        )