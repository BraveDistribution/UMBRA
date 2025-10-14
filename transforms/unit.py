"""
Custom MONAI-compatible transforms.
"""

from __future__ import annotations
from typing import Sequence

import torch
import numpy as np
from monai.transforms.transform import MapTransform, Randomizable
from monai.utils.enums import TransformBackends
from monai.data.meta_tensor import MetaTensor

from utils.misc import ensure_tuple_dim

class CreateRandomMaskd(MapTransform, Randomizable):
    """
    Create a random 3D mask for masked autoencoder training.

    Note:
        Choose a mask patch size that is large enough to avoid leakage of unmasked 
        tokens in the bottleneck -> mask the bottleneck and interpolate back to voxel space.

        This is not done automatically here.
    """
    backend = [TransformBackends.TORCH]

    def __init__(
        self,
        keys: Sequence[str] = ("img",),
        mask_key: str = "mask", 
        mask_ratio: float | Sequence[float] = 0.6,
        mask_patch_size: int = 4,
        allow_missing_keys: bool = False
    ) -> None:
        """
        Args:
            keys: Keys to generate masks for (typically 'img')
            mask_key: Key to store the generated mask
            mask_ratio: Fraction of voxels to mask (0.0 to 1.0) or sequence of fractions (e.g. (0.6, 0.75))
                to randomly sample from.
            mask_patch_size: Size of mask patches (cube side length)
            allow_missing_keys: Whether to allow missing keys
        """
        super().__init__(keys, allow_missing_keys)

        self.mask_key = mask_key
        self.mask_ratio_range = ensure_tuple_dim(mask_ratio, 2)
        if self.mask_ratio_range[0] > self.mask_ratio_range[1]:
            raise ValueError("mask_ratio[0] should be <= mask_ratio[1]")
        self.mask_patch_size = mask_patch_size
    
    def randomize(self, img_shape):
        # Calculate number of patches in each dimension
        d1, d2, d3 = img_shape[-3:]
        n_patches = (np.array([d1, d2, d3]) + self.mask_patch_size - 1) // self.mask_patch_size
        total_patches = np.prod(n_patches)
        mask_ratio = self.R.uniform(self.mask_ratio_range)

        # Sample which patches to mask
        n_masked = int(total_patches * mask_ratio)
        patch_mask = torch.ones(int(total_patches), dtype=torch.bool)

        idx = torch.as_tensor(
            self.R.choice(int(total_patches), n_masked, replace=False)
        )
        patch_mask[idx] = False
        self._patch_mask = patch_mask.view(*map(int, n_patches))
    
    def __call__(self, data):
        d = dict(data)
        
        for key in self.key_iterator(d):
            img = d[key]
            img_shape = img.shape
            try:
                d1, d2, d3 = img_shape[-3:]
            except Exception as e:
                raise ValueError(f"CreateRandomMaskd expects a 3-D volume, got {img_shape}") from e
            
            self.randomize(img_shape)

            # Vectorised up-sampling back to voxel space
            mask = self._patch_mask.repeat_interleave(self.mask_patch_size, 0) \
                                 .repeat_interleave(self.mask_patch_size, 1) \
                                 .repeat_interleave(self.mask_patch_size, 2)                  

            # Crop in case spatial dims are not multiples of patch size
            mask = mask[:d1, :d2, :d3]

            # Get original dims
            while mask.ndim < img.ndim:
                mask = mask.unsqueeze(0)
 
            # Keep metadata
            if isinstance(img, MetaTensor):
                d[self.mask_key] = MetaTensor(mask, meta=img.meta)
            else:
                d[self.mask_key] = mask
        return d


class GetReconstructionTargetd(MapTransform):
    """
    Get a copy of the image as reconstruction target before intensity augmentations.
    This creates a 'recon' key with the current image data.
    """
    def __init__(
        self, 
        keys: Sequence[str] | str = "img", 
        recon_key: str = "recon",
        allow_missing_keys: bool = False
    ) -> None:
        super().__init__(keys, allow_missing_keys)
        self.recon_key = recon_key
    
    def __call__(self, data):
        d = dict(data)
        for key in self.key_iterator(d):
            d[self.recon_key] = d[key].clone()
        return d