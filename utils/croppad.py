"""Cropping and padding utilities."""

from __future__ import annotations

__all__ = [
    "random_crop",
]

from typing import Tuple

import numpy as np
from numpy.typing import NDArray

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