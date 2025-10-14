"""
Random masking augmentation for self-supervised learning.

This module provides utilities for applying random masking to input tensors,
commonly used in masked autoencoding (MAE) approaches.
"""

import torch

from utils.masking import generate_random_mask

def random_mask(
    x: torch.Tensor,
    mask_ratio: float,
    mask_patch_size: int,
    mask_token: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Apply random masking to input tensor.

    Args:
        x: Input tensor of shape (B, C, H, W) or (B, C, H, W, D)
        mask_ratio: Ratio of patches to mask (0.0 to 1.0)
        mask_patch_size: Size of each patch for masking
        mask_token: Value to use for masked regions (default: 0.0)

    Returns:
        Tuple of (masked_tensor, mask) where:
        - masked_tensor: Input with masked regions set to mask_token
        - mask: Boolean tensor indicating masked regions (True = masked)

    Example:
        >>> x = torch.randn(2, 1, 96, 96, 96)
        >>> masked_x, mask = random_mask(x, mask_ratio=0.6, mask_patch_size=4)
    """
    mask = generate_random_mask(x, mask_ratio, mask_patch_size, out_type=bool)

    # Validate mask type
    assert mask.dtype == torch.bool, f"Expected bool tensor, got {mask.dtype}"

    # Apply mask
    x[mask] = mask_token

    return x, mask
