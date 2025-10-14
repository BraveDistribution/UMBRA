"""
MONAI-compatible composed transforms.
"""

from __future__ import annotations
from typing import Sequence, Union

# pyright: reportPrivateImportUsage=false
from monai.transforms import (
    Compose,
    EnsureChannelFirstd,
    OneOf,
    RandFlipd,
    RandAffined,
    RandAdjustContrastd,
    RandBiasFieldd,
    RandGaussianNoised,
    RandGaussianSmoothd,
    RandGibbsNoised,
    RandScaleIntensityFixedMeand,
    RandSimulateLowResolutiond,
    RandSpatialCropd,
    SpatialPadd,
    ToTensord,
)

from transforms.unit import (
    GetReconstructionTargetd, 
    CreateRandomMaskd,
)

def get_mae_transforms(
    keys: Sequence[str] = ("volume",),
    patch_size: Union[int, Sequence[int]] = 96,
    mask_ratio: Union[float, Sequence[float]] = (0.6, 0.75),
    mask_patch_size: int = 4,
    val_mode: bool = False,
) -> Compose:
    """
    Get MAE transforms for training.
    """
    # Default I/O
    transforms = [
        ToTensord(keys=keys),
        EnsureChannelFirstd(keys=keys),
    ]

    # Spatial augmentations
    if not val_mode:
        transforms.extend([
            RandFlipd(keys=keys, prob=0.5, spatial_axis=0),
            RandFlipd(keys=keys, prob=0.5, spatial_axis=1),
            RandAffined(keys=keys, rotate_range=(0.3, 0.3, 0.3),
                        scale_range=(0.1, 0.1, 0.1), shear_range=(0.3, 0.3, 0.3),
                        mode='bilinear', padding_mode='border', prob=1.0),
        ])

    # Get reconstruction target
    transforms.extend([
        GetReconstructionTargetd(keys=keys, recon_key="recon"),
    ])

    # Intensity augmentations
    if not val_mode:
        transforms.extend([
            RandScaleIntensityFixedMeand(keys=keys, factors=0.1, prob=0.8),
            RandGaussianNoised(keys=keys, std=0.01, prob=0.3),
        ])
        # Simulate artefacts
        transforms.extend([
            OneOf(transforms=[
                RandGaussianSmoothd(keys=keys, sigma_x=(0.5, 1.0), sigma_y=(0.5, 1.0),
                                    sigma_z=(0.5, 1.0), prob=0.7),
                RandBiasFieldd(keys=keys, coeff_range=(0.0, 0.05), prob=0.7),
                RandGibbsNoised(keys=keys, alpha=(0.2, 0.4), prob=0.7),
            ], weights=[1.0, 1.0, 1.0]),
        ])
        # Simulate different acquisitions
        transforms.extend([
            OneOf(transforms=[
                RandAdjustContrastd(keys=keys, gamma=(0.9, 1.1), prob=1.0),
                RandSimulateLowResolutiond(keys=keys, prob=0.5, zoom_range=(0.8, 1.0)),
            ], weights=[1.0, 1.0]),
        ])
    
    # Get patch size
    transforms.extend([
        SpatialPadd(keys=keys, spatial_size=patch_size),
        RandSpatialCropd(keys=keys, roi_size=patch_size),
    ])

    # Get mask
    transforms.extend([
        CreateRandomMaskd(keys=keys, mask_ratio=mask_ratio, mask_patch_size=mask_patch_size),
    ])

    return Compose(transforms)