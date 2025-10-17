"""Cropping and padding utilities."""

from __future__ import annotations

__all__ = [
    "random_crop",
    "shared_random_crop",
]

from typing import Tuple, Sequence, Union

import numpy as np
from numpy.typing import NDArray
import torch
import torch.nn.functional as F

from utils.misc import ensure_tuple_dim

def random_crop(
        volume: NDArray[np.float32],
        patch_size: Tuple[int, int, int],
    ) -> NDArray[np.float32]:
        """Apply random crop to ensure uniform size."""
        import random
        pd, ph, pw = patch_size
        D, H, W = volume.shape[-3:]

        # Random crop coordinates
        sd = 0 if D <= pd else random.randint(0, D - pd)
        sh = 0 if H <= ph else random.randint(0, H - ph)
        sw = 0 if W <= pw else random.randint(0, W - pw)

        # Apply crop
        cropped = volume[..., sd : sd + pd, sh : sh + ph, sw : sw + pw]

        # If any dim is smaller than patch, pad
        if cropped.shape[-3] < pd or cropped.shape[-2] < ph or cropped.shape[-1] < pw:
            Cd, Ch, Cw = cropped.shape[-3:]
            pad_d = max(0, pd - Cd)
            pad_h = max(0, ph - Ch)
            pad_w = max(0, pw - Cw)

            cropped = np.pad(
                cropped,
                ((0, 0), (0, pad_d), (0, pad_h), (0, pad_w)),
                mode="constant",
                constant_values=0,
            )

        return cropped

def shared_random_crop(
        v1: NDArray[np.float32],
        v2: NDArray[np.float32],
        patch_size: Tuple[int, int, int],
    ) -> Tuple[NDArray[np.float32], NDArray[np.float32]]:
        """
        Apply the same random crop to both volumes so aligned.
        v1, v2: (C, D, H, W) arrays on the same grid.
        patch_size: tuple (pd, ph, pw) for depth, height, width.
        """
        import random
        pd: int
        ph: int
        pw: int
        pd, ph, pw = patch_size
        D: int
        H: int
        W: int
        D, H, W = v1.shape[-3:]

        # Random crop coordinates
        sd: int = 0 if D <= pd else random.randint(0, D - pd)
        sh: int = 0 if H <= ph else random.randint(0, H - ph)
        sw: int = 0 if W <= pw else random.randint(0, W - pw)

        # Apply crop to both
        v1c: NDArray[np.float32] = v1[..., sd : sd + pd, sh : sh + ph, sw : sw + pw]
        v2c: NDArray[np.float32] = v2[..., sd : sd + pd, sh : sh + ph, sw : sw + pw]

        # If any dim is smaller than patch, pad both identically
        def _pad_to_patch(x: NDArray[np.float32]) -> NDArray[np.float32]:
            Cd: int
            Ch: int
            Cw: int
            Cd, Ch, Cw = x.shape[-3:]
            pad_d: int = max(0, pd - Cd)
            pad_h: int = max(0, ph - Ch)
            pad_w: int = max(0, pw - Cw)
            if pad_d or pad_h or pad_w:
                x = np.pad(
                    x,
                    ((0, 0), (0, pad_d), (0, pad_h), (0, pad_w)),
                    mode="constant",
                    constant_values=0,
                )
            return x

        v1c = _pad_to_patch(v1c)
        v2c = _pad_to_patch(v2c)

        return v1c, v2c

def upsample_to_3d(
    x: torch.Tensor, 
    ref: torch.Tensor, 
    mode: str = "trilinear"
) -> torch.Tensor:
    """
    Upsample x to the same spatial dimensions as ref.
    """
    return F.interpolate(
        x, size=ref.shape[-3:], mode=mode, align_corners=False
    )

def voxel_shuffle_3d(x: torch.Tensor, r: Union[Sequence[int], int]) -> torch.Tensor:
    """
    Voxel shuffle operation for 3D tensors.

    Can be used to convert a tesnor in the patch grid back to the original
    voxel grid.

    Args:
        x: Input tensor of shape (B, C*r1*r2*r3, D, H, W)
        r: Upscale factors per spatial dim; use int for isotropic

    Returns:
        Output tensor of shape (B, C, D*r1, H*r2, W*r3)
    """
    # x: (B, C*r1*r2*r3, D, H, W) -> (B, C, D*r1, H*r2, W*r3)
    B, C_r1_r2_r3, D, H, W = x.shape
    r1, r2, r3 = ensure_tuple_dim(r, 3)
    assert C_r1_r2_r3 % (r1 * r2 * r3) == 0, "Channel dim must be divisible by up1*up2*up3"
    C = C_r1_r2_r3 // (r1 * r2 * r3)
    x = x.view(B, C, r1, r2, r3, D, H, W)
    x = x.permute(0, 1, 5, 2, 6, 3, 7, 4).contiguous()
    return x.view(B, C, D * r1, H * r2, W * r3)
