"""Utility functions for Pytorch networks."""

from __future__ import annotations

__all__ = [
    "init_swin_v2",
    "swap_in_to_gn",
]

from typing import Any, Sequence, Dict, Union, Optional, List, Tuple
from typing import TYPE_CHECKING
from pathlib import Path
import math

import torch
import torch.nn as nn

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

def get_optimizer_lr(optimizers: List[optim.Optimizer]) -> Dict[str, float]:
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

def load_param_group_from_ckpt(
    model_instance: nn.Module,
    *,
    checkpoint_path: Path,
    select_prefixes: Optional[Union[str, Sequence[str]]] = None,
    rename_map: Optional[Dict[str, str]] = None,
    strict: bool = False,
    torch_load_kwargs: Optional[Dict[str, Any]] = None,
) -> tuple[nn.Module, Dict[str, Any]]:
    """
    Adapted from: https://github.com/MaastrichtU-CDS/anyBrainer/

    Load (optionally a subset of) parameters from a checkpoint into a module,
    with optional prefix-based key renaming.

    Args:
        model_instance: The module to load parameters into.
        checkpoint_path: The path to the checkpoint file.
        select_prefixes: A list of prefixes to select parameters from.
        rename_map: A dictionary of old prefixes to new prefixes. 
        strict: Whether to raise an error if there are missing or unexpected keys.
        torch_load_kwargs: Additional keyword arguments to pass to `torch.load`.

    Returns:
        model_instance: The loaded module.
        stats: A dictionary of statistics.
    
    Raises:
        FileNotFoundError: if ``checkpoint_path`` does not exist.
        TypeError: if the loaded checkpoint does not contain a dict-like state dict.
    """
    kw = torch_load_kwargs or {}
    ckpt = torch.load(str(checkpoint_path), **kw)
    state_dict = ckpt.get("state_dict", ckpt)
    if not isinstance(state_dict, dict):
        raise TypeError("No dict-like 'state_dict' in checkpoint.")

    # selection
    if select_prefixes:
        if isinstance(select_prefixes, str):
            select_prefixes = [select_prefixes]
        selected = {k: v for k, v in state_dict.items() if any(k.startswith(p) for p in select_prefixes)}
        ignored = [k for k in state_dict if k not in selected]
    else:
        selected = dict(state_dict)
        ignored = []

    # renaming
    to_load = {}
    if rename_map:
        for k, v in selected.items():
            new_k = k
            for old, new in rename_map.items():
                if k.startswith(old):
                    new_k = new + k[len(old):]
                    break
            to_load[new_k] = v
    else:
        to_load = selected

    result = model_instance.load_state_dict(to_load, strict=strict)
    if hasattr(result, "missing_keys"):
        missing_keys = list(result.missing_keys)
        unexpected_keys = list(result.unexpected_keys)
    elif isinstance(result, tuple) and len(result) == 2:
        missing_keys, unexpected_keys = list(result[0]), list(result[1])
    else:
        missing_keys, unexpected_keys = [], []

    stats = {
        "loaded_keys": list(to_load.keys()),
        "ignored_keys": ignored,
        "missing_keys": missing_keys,
        "unexpected_keys": unexpected_keys,
    }
    return model_instance, stats

def split_decay_no_decay(
    module: nn.Module,
    *,
    extra_no_decay_modules: Tuple[type, ...] = (),
    extra_no_decay_names: Tuple[str, ...] = (),
) -> Tuple[List[nn.Parameter], List[nn.Parameter]]:
    """
    Return (decay_params, no_decay_params) for AdamW-style grouping.

    Rules:
      - no decay: biases and affine params of normalization layers
      - decay: everything else (Linear/Conv/Attention/MLP weights, patch/pos embeds, etc.)
    You can extend "no decay" via `extra_no_decay_modules` or `extra_no_decay_names` 
    (substring match on param name).
    """
    norm_types = (
        nn.LayerNorm, nn.GroupNorm,
        nn.InstanceNorm1d, nn.InstanceNorm2d, nn.InstanceNorm3d,
        nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.SyncBatchNorm,
    ) + tuple(extra_no_decay_modules)

    decay, no_decay = [], []
    seen: set = set()

    for m in module.modules():
        for name, p in m.named_parameters(recurse=False):
            if p is None or not p.requires_grad:
                continue
            pid = id(p)
            if pid in seen:
                continue
            seen.add(pid)

            # biases -> no decay
            if name.endswith("bias") or name == "bias":
                no_decay.append(p)
                continue

            # norm affine params -> no decay
            if isinstance(m, norm_types):
                # LayerNorm/GroupNorm/InstanceNorm typically have weight/bias attributes
                no_decay.append(p)
                continue

            # extra no-decay name patterns (substring match on local name)
            if extra_no_decay_names and any(k in name for k in extra_no_decay_names):
                no_decay.append(p)
                continue

            # default: decay
            decay.append(p)

    return decay, no_decay