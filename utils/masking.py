"""
Masking utilities for self-supervised learning.

This module provides core masking functionality for generating random masks
on 2D and 3D tensors, commonly used in masked autoencoding approaches.
"""

from __future__ import annotations

__all__ = [
    "generate_random_mask",
]

import torch

def generate_random_mask(
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
