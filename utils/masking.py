"""
Masking utilities for self-supervised learning.

This module provides core masking functionality for generating random masks
on 2D and 3D tensors, commonly used in masked autoencoding approaches.
"""

from __future__ import annotations

__all__ = [
    "generate_random_mask_conv",
    "generate_random_mask_vit",
]

from typing import Sequence, Tuple, Union, Optional, Literal
import torch
import torch.nn.functional as F

from utils.misc import ensure_tuple_dim

# ----------ConvNet-compatible masking-----------#
def _ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b

def up_to_voxel_space(
    mask_finest_grid: torch.Tensor, 
    original_spatial: Sequence[int],
    patch_size: Sequence[int]
) -> torch.Tensor:
    """
    Expand a finest-token-grid mask to voxel space by repeating each token cell
    by the patch size along each spatial axis, then trim to the original size.

    Args:
        mask_finest_grid: (B, Df,Hf,Wf) bool or (B, Hf,Wf) bool
        original_spatial: (D,H,W) or (H,W) (pre-padding)
        patch_size: (pD,pH,pW) or (pH,pW)

    Returns:
        voxel_mask: mask at voxel space (B, 1, D, H, W) or (B, 1, H, W) bool
    """
    ps = tuple(patch_size)

    if mask_finest_grid.dim() == 4: # 3D: (B,Df,Hf,Wf)
        pD, pH, pW = ps
        vm = (mask_finest_grid.unsqueeze(1) # (B,1,Df,Hf,Wf)
              .repeat_interleave(pD, dim=-3)
              .repeat_interleave(pH, dim=-2)
              .repeat_interleave(pW, dim=-1))
        D, H, W = original_spatial
        vm = vm[..., :D, :H, :W] # trim padding
    elif mask_finest_grid.dim() == 3: # 2D: (B,Hf,Wf)
        pH, pW = ps
        vm = (mask_finest_grid.unsqueeze(1) # (B,1,Hf,Wf)
               .repeat_interleave(pH, dim=-2)
               .repeat_interleave(pW, dim=-1))
        H, W = original_spatial
        vm = vm[..., :H, :W] # trim padding
    else:
        raise ValueError("`mask_finest_grid` must be (B,Df,Hf,Wf) or (B,Hf,Wf).")
    
    return vm.bool()

@torch.no_grad()
def generate_random_mask_conv(
    x_spatial: Sequence[int],
    patch_size: Sequence[int],
    num_downsamples: int,
    batch_size: int,
    mask_ratio: Union[float, Sequence[float]],
    *,
    return_kind: Literal["flat", "grid", "voxel"] = "flat",
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """
    Create a hierarchical mask suitable for ConvNets-based MAE:
      1) compute finest token grid to match output of `PatchEmbed`
      2) sample masked tokens on the bottleneck grid
      3) upsample (nearest) to the finest token grid

    Args:
        x_spatial: Spatial dimensions of input tensor (D,H,W) or (H,W) (convention for ConvNets)
        patch_size: Patch size (pD,pH,pW) or (pH,pW)
        num_downsamples: Number of downsamples (typically len(depths))
        batch_size: Batch size
        mask_ratio: Mask ratio (0.0 to 1.0) or sequence of mask ratios (e.g. (0.6, 0.75))
            to uniformly randomly sample from.
        return_kind: Type of mask to return ("flat", "grid", "voxel")
        device: Device to create the mask on

    Returns:
      - "flat":  (B, N_f) boolean mask for patch-embed token injection
      - "grid":  (B, Df, Hf, Wf) boolean mask at finest token grid
      - "voxel": (B, 1, D, H, W) boolean voxel-space mask (nearest repeat)
    """
    # Normalize shapes
    spatial = tuple(x_spatial)
    ps = tuple(patch_size)
    mask_ratio = ensure_tuple_dim(mask_ratio, 2)
    assert len(spatial) in [2, 3], f"Expected 2D or 3D input, got {len(spatial)}D"
    assert len(ps) == len(spatial), "patch_size must match spatial dimensions"
    assert mask_ratio[0] <= mask_ratio[1], "mask_ratio[0] should be <= mask_ratio[1]"

    if device is None:
        device = torch.device("cpu")

    # Get finest (after patch-embed) and bottleneck (after downsamples) token grids
    finest = tuple(_ceil_div(s, p) for s, p in zip(spatial, ps))
    bottleneck = tuple(_ceil_div(f, 2 ** num_downsamples) for f in finest)

    # num tokens
    N_b = int(torch.tensor(bottleneck).prod().item())
    N_f = int(torch.tensor(finest).prod().item())
    
    # Per-sample masking at bottleneck with unique mask ratios
    ratios = torch.empty(batch_size, device=device).uniform_(
        float(mask_ratio[0]), float(mask_ratio[1])
    )
    mask_b = torch.zeros(batch_size, N_b, device=device, dtype=torch.bool)
    for b in range(batch_size):
        k = int(round(ratios[b].item() * N_b))
        if k > 0:
            idx = torch.randperm(N_b, device=device)[:k]
            mask_b[b, idx] = True

    # Upsample mask to finest grid
    # Up to finest token grid
    if len(spatial) == 3:
        Db, Hb, Wb = bottleneck
        Df, Hf, Wf = finest
        mb = mask_b.view(batch_size, 1, Db, Hb, Wb).float()
        mf = F.interpolate(mb, size=(Df, Hf, Wf), mode="nearest").squeeze(1).bool()
    else:
        Hb, Wb = bottleneck
        Hf, Wf = finest
        mb = mask_b.view(batch_size, 1, Hb, Wb).float()
        mf = F.interpolate(mb, size=(Hf, Wf), mode="nearest").squeeze(1).bool()

    # Return mask of requested shape
    if return_kind == "grid":
        return mf
    if return_kind == "flat":
        return mf.view(batch_size, N_f)

    return up_to_voxel_space(mf, spatial, ps)


# ---------Original vanilla ViT implementation; not applicable for ConvNets---------#
def generate_random_mask_vit(
    x: torch.Tensor,
    mask_ratio: float,
    patch_size: int,
    out_type: type = int,
) -> torch.Tensor:
    """
    Generate a random mask for input tensor using patch-based masking.

    Args:
        x: Input tensor of shape (B, C, H, W) for 2D or (B, C, H, W, D) for 3D
        mask_ratio: Ratio of patches to mask (0.0 to 1.0)
        patch_size: Size of each square/cubic patch
        out_type: Output type for mask (int or bool)

    Returns:
        Binary mask tensor where 1/True indicates masked regions and
        0/False indicates kept regions. Shape matches input x.

    Raises:
        AssertionError: If input dimensions are invalid or not divisible by
            patch_size

    Example:
        >>> x = torch.randn(2, 1, 96, 96, 96)
        >>> mask = generate_random_mask(x, mask_ratio=0.75, patch_size=4)
        >>> mask.shape
        torch.Size([2, 1, 96, 96, 96])
    """
    # Determine dimensionality (2D or 3D)
    dim = len(x.shape) - 2
    assert dim in [2, 3], f"Expected 2D or 3D input, got {dim}D"

    # Validate spatial dimensions are divisible by patch_size
    for i in range(2, len(x.shape)):
        assert x.shape[i] % patch_size == 0, (
            f"Spatial dimension {i} (size {x.shape[i]}) must be divisible "
            f"by patch_size {patch_size}"
        )

    # Generate 1D mask and reshape to match input dimensions
    mask = _generate_1d_mask(x, mask_ratio, patch_size, out_type)
    mask = _reshape_to_dim(mask, x.shape, patch_size)

    # Upsample mask to match original spatial resolution
    up_mask = _upsample_mask(mask, patch_size)

    return up_mask


def _generate_1d_mask(
    x: torch.Tensor, mask_ratio: float, patch_size: int, out_type: type
) -> torch.Tensor:
    """
    Generate 1D flattened mask for patches.

    Args:
        x: Input tensor
        mask_ratio: Ratio of patches to mask
        patch_size: Size of each patch
        out_type: Output type (int or bool)

    Returns:
        1D mask of shape (B, num_patches) where 0/False is keep, 1/True is
        remove
    """
    assert x.shape[1] in [1, 3], f"Expected 1 or 3 channels, got {x.shape[1]}"
    assert out_type in [int, bool], "out_type must be int or bool"

    batch_size = x.shape[0]

    # Calculate total number of patches across all spatial dimensions
    num_patches = 1
    for i in range(2, len(x.shape)):
        num_patches *= x.shape[i] // patch_size

    # Calculate number of patches to keep
    len_keep = int(num_patches * (1 - mask_ratio))

    # Generate random noise for shuffling
    noise = torch.randn(batch_size, num_patches, device=x.device)

    # Create shuffled indices
    ids_shuffle = torch.argsort(noise, dim=1)
    ids_restore = torch.argsort(ids_shuffle, dim=1)

    # Generate binary mask: 0 is keep, 1 is remove
    mask = torch.ones([batch_size, num_patches], device=x.device)
    mask[:, :len_keep] = 0

    # Unshuffle to get final mask
    mask = torch.gather(mask, dim=1, index=ids_restore)

    # Convert to requested type
    if out_type == bool:
        mask = mask.bool()
    elif out_type == int:
        mask = mask.int()

    return mask


def _reshape_to_dim(
    mask: torch.Tensor, original_shape: tuple, patch_size: int
) -> torch.Tensor:
    """
    Reshape 1D mask to match spatial dimensions.

    Args:
        mask: 1D mask of shape (B, num_patches)
        original_shape: Original input shape
        patch_size: Size of each patch

    Returns:
        Reshaped mask of shape (B, H_p, W_p) or (B, H_p, W_p, D_p) where
        *_p denotes patch dimensions
    """
    dim = len(original_shape) - 2
    assert dim in [2, 3], f"Expected 2D or 3D, got {dim}D"
    assert len(mask.shape) == 2, "Mask should be 2D (batch, num_patches)"

    if dim == 2:
        h_patches = original_shape[2] // patch_size
        w_patches = original_shape[3] // patch_size
        return mask.reshape(-1, h_patches, w_patches)
    else:  # dim == 3
        h_patches = original_shape[2] // patch_size
        w_patches = original_shape[3] // patch_size
        d_patches = original_shape[4] // patch_size
        return mask.reshape(-1, h_patches, w_patches, d_patches)


def _upsample_mask(mask: torch.Tensor, scale: int) -> torch.Tensor:
    """
    Upsample mask by repeating values.

    Args:
        mask: Mask of shape (B, H, W) or (B, H, W, D)
        scale: Upsampling scale factor (patch_size)

    Returns:
        Upsampled mask with channel dimension added:
        (B, 1, H*scale, W*scale) or (B, 1, H*scale, W*scale, D*scale)
    """
    assert scale > 0, "Scale must be positive"
    assert len(mask.shape) in [3, 4], f"Expected 3D or 4D mask, got {len(mask.shape)}D"

    if len(mask.shape) == 3:
        # 2D case: (B, H, W) -> (B, H*scale, W*scale)
        mask = mask.repeat_interleave(scale, dim=1).repeat_interleave(scale, dim=2)
    else:
        # 3D case: (B, H, W, D) -> (B, H*scale, W*scale, D*scale)
        mask = (
            mask.repeat_interleave(scale, dim=1)
            .repeat_interleave(scale, dim=2)
            .repeat_interleave(scale, dim=3)
        )

    # Add channel dimension
    return mask.unsqueeze(1)
