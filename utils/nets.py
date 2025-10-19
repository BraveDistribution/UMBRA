"""Utility functions for Pytorch networks."""

from __future__ import annotations

__all__ = [
    "init_swin_v2",
    "swap_in_to_gn",
]

from typing import TYPE_CHECKING
import math

import torch
import torch.nn as nn
try:
    from lightning.pytorch.utilities import rank_zero_only
except ImportError:
    try:
        from pytorch_lightning.utilities import rank_zero_only
    except ImportError:
        # Fallback for when neither import works
        def rank_zero_only(func):
            """Dummy rank_zero_only decorator that does nothing."""
            return func

if TYPE_CHECKING:
    import torch.optim as optim

def _trunc_normal_(tensor: torch.Tensor, mean: float = 0., std: float = 1.,
                   a: float = -2., b: float = 2.) -> torch.Tensor:
    """
    Fills `tensor` with values drawn from a truncated N(mean, std) distribution.
    Modifies `tensor` in-place.
    """
    def norm_cdf(x: float) -> float:
        """Cumulative distribution function for the standard normal."""
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    l = norm_cdf((a - mean) / std)
    u = norm_cdf((b - mean) / std)

    with torch.no_grad():
        tensor.uniform_(2 * l - 1, 2 * u - 1)
        tensor.erfinv_()
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)
        tensor.clamp_(min=a, max=b)

    return tensor

def init_swin_v2(
    model: nn.Module,
    *,
    proj_std: float = 0.02,
    conv_mode: str = "fan_out",
    zero_init_residual: bool = False
) -> None:
    """
    Weight initialisation for a Swinv2-style ViT that contains
    linear projection layers and (optionally 3 x 3) residual convolutions
    before each Swinv2 block.

    Args:
        model : nn.Module
            Complete Swinv2 ViT model (or sub-module).
        proj_std : float, default 0.02
            Std for truncated-normal init of all nn.Linear weights (Swinv2 default).
        conv_mode : {"fan_out", "fan_in"}, default "fan_out"
            Kaiming mode for convolutions. "fan_out" is common in modern conv nets.
        zero_init_residual : bool, default False
            If True, the *last* conv in every residual branch is zero-initialised,
            encouraging the residual path to start as identity.
    """
    # helper to decide whether a conv is "last" in a residual branch
    def _is_last_residual_conv(m: nn.Conv2d | nn.Conv3d) -> bool:
        # convention: last conv in residual conv stack ends with ".2"
        # e.g. 'residual_conv.2.weight'
        return m.weight.shape[0] == m.weight.shape[1] and m.kernel_size == (3, 3)

    for _, m in model.named_modules():
        # Convolution layers
        if isinstance(m, (nn.Conv2d, nn.Conv3d)):
            if zero_init_residual and _is_last_residual_conv(m):
                nn.init.zeros_(m.weight)
            else:
                nn.init.kaiming_normal_(m.weight, mode=conv_mode, nonlinearity="relu") # type: ignore
            if m.bias is not None:
                nn.init.zeros_(m.bias)

        # Linear (projection) layers
        elif isinstance(m, nn.Linear):
            _trunc_normal_(m.weight, std=proj_std)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

        # Normalization layers
        elif isinstance(m, (nn.LayerNorm, nn.GroupNorm, nn.BatchNorm1d,
                            nn.BatchNorm2d, nn.BatchNorm3d)):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

def swap_in_to_gn(module: nn.Module, groups: int = 8) -> None:
    """
    Swap InstanceNorm layers to GroupNorm layers.
    
    Args:
        module: nn.Module
            Complete network (or sub-module).
        groups: int, default 8
            Number of groups for GroupNorm.
    """
    for name, child in list(module.named_children()):
        if isinstance(child, (nn.InstanceNorm2d, nn.InstanceNorm3d)):
            C = child.num_features
            gn = nn.GroupNorm(num_groups=min(groups, C), num_channels=C,
                              eps=child.eps, affine=True)
            setattr(module, name, gn)
        else:
            swap_in_to_gn(child, groups)

def get_optimizer_lr(optimizers: list[optim.Optimizer]) -> dict[str, float]:
    """Get optimizer learning rates."""
    return {
        f"train/lr/opt{i}_group{j}": group["lr"]
        for i, opt in enumerate(optimizers)
        for j, group in enumerate(opt.param_groups)
    }

def get_total_grad_norm(model: nn.Module) -> torch.Tensor:
    """Get total gradient norm."""
    total_norm = torch.norm(
        torch.stack([
            p.grad.detach().norm(2)
            for p in model.parameters()
            if p.grad is not None
        ]), p=2,
    )
    return total_norm