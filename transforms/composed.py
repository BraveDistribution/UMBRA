"""
MONAI-compatible composed transforms.
"""

from __future__ import annotations

__all__ = [
    "get_mae_transforms",
    "get_contrastive_transforms",
]

from typing import Callable, Dict, Sequence, Union
from typing import cast

from numpy.typing import NDArray
from torch import Tensor
# pyright: reportPrivateImportUsage=false
from monai.transforms import (
    CenterSpatialCropd,
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
)

def get_mae_transforms(
    keys: Sequence[str] = ("volume",),
    input_size: Union[int, Sequence[int]] = 96,
    val_mode: bool = False,
) -> Callable[[Dict[str, NDArray]], Dict[str, Tensor]]:
    """
    Get MAE transforms for training or validation.

    Saves reconstruction target as `keys[0]_recon` key. 

    Args:
        keys: Keys to apply the transforms to.
        input_size: Target size of returned tensors.
        val_mode: Whether to use validation mode.

    Returns:
        Callable object that applies the transforms.
    """
    # Default I/O
    transforms = [
        ToTensord(keys=keys),
        EnsureChannelFirstd(keys=keys, channel_dim=0),
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
        GetReconstructionTargetd(keys=keys, recon_key=f"{keys[0]}_recon"),
    ])

    # Intensity augmentations
    if not val_mode:
        transforms.extend([
            RandScaleIntensityFixedMeand(keys=keys, factors=0.1, prob=0.8),
            RandGaussianNoised(keys=keys, std=0.01, prob=0.3),
        ])
        # Simulate artifacts
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
    
    # Pad and crop to match input size for volume(s) and target
    keys_with_recon = [f"{keys[0]}_recon", *keys]
    transforms.extend([
        SpatialPadd(keys=keys_with_recon, spatial_size=input_size),
    ])
    if not val_mode:
        transforms.extend([
            RandSpatialCropd(keys=keys_with_recon, roi_size=input_size),
        ])
    else:
        transforms.extend([
            CenterSpatialCropd(keys=keys_with_recon, roi_size=input_size),
        ])

    return cast(Callable[[Dict[str, NDArray]], Dict[str, Tensor]], Compose(transforms))

def get_contrastive_transforms(
    keys: Sequence[str] = ("vol1", "vol2"),
    input_size: Union[int, Sequence[int]] = 96,
    conservative_mode: bool = True,
    val_mode: bool = False,
    recon: bool = False,
) -> Callable[[Dict[str, NDArray]], Dict[str, Tensor]]:
    """
    Get contrastive transforms for training or validation.

    Args:
        keys: Keys to apply the transforms to.
        input_size: Target size of returned tensors.
        conservative_mode: Whether to apply same spatial augmentations  and cropping 
            to both volumes. If False, each volume is augmented and cropped independently.
        val_mode: Whether to use validation mode.
        recon: Whether to save reconstruction target as `keys[0]_recon` and `keys[1]_recon` keys.

    Returns:
        Compose object with the transforms.
    """
    # Default I/O
    transforms = [
        ToTensord(keys=keys),
        EnsureChannelFirstd(keys=keys, channel_dim=0),
    ]

    # Augmentations
    if not val_mode:
        # Spatial
        if conservative_mode:
            transforms.extend([
                RandFlipd(keys=keys, prob=0.5, spatial_axis=0),
            ])
        else:
            transforms.extend([
                RandFlipd(keys=keys[0], prob=0.5, spatial_axis=0),
                RandFlipd(keys=keys[0], prob=0.5, spatial_axis=1),
                RandFlipd(keys=keys[1], prob=0.5, spatial_axis=0),
                RandFlipd(keys=keys[1], prob=0.5, spatial_axis=1),
            ])
        transforms.extend([
            RandAffined(keys=keys[0], rotate_range=(0.3, 0.3, 0.3),
                        scale_range=(0.1, 0.1, 0.1), shear_range=(0.3, 0.3, 0.3),
                        mode='bilinear', padding_mode='border', prob=1.0),
            RandAffined(keys=keys[1], rotate_range=(0.3, 0.3, 0.3),
                        scale_range=(0.1, 0.1, 0.1), shear_range=(0.3, 0.3, 0.3),
                        mode='bilinear', padding_mode='border', prob=1.0),
        ])

        if recon:
            transforms.extend([
                GetReconstructionTargetd(keys=keys[0], recon_key=f"{keys[0]}_recon"),
                GetReconstructionTargetd(keys=keys[1], recon_key=f"{keys[1]}_recon"),
            ])

        # Intensity
        transforms.extend([
            RandScaleIntensityFixedMeand(keys=keys[0], factors=0.1, prob=0.8),
            RandScaleIntensityFixedMeand(keys=keys[1], factors=0.1, prob=0.8),
            RandGaussianNoised(keys=keys[0], std=0.01, prob=0.3),
            RandGaussianNoised(keys=keys[1], std=0.01, prob=0.3),
        ])
        # Simulate artifacts
        transforms.extend([
            OneOf(transforms=[
                RandGaussianSmoothd(keys=keys[0], sigma_x=(0.5, 1.0), sigma_y=(0.5, 1.0),
                                    sigma_z=(0.5, 1.0), prob=0.7),
                RandBiasFieldd(keys=keys[0], coeff_range=(0.0, 0.05), prob=0.7),
                RandGibbsNoised(keys=keys[0], alpha=(0.2, 0.4), prob=0.7),
            ], weights=[1.0, 1.0, 1.0]),
            OneOf(transforms=[
                RandGaussianSmoothd(keys=keys[1], sigma_x=(0.5, 1.0), sigma_y=(0.5, 1.0),
                                    sigma_z=(0.5, 1.0), prob=0.7),
                RandBiasFieldd(keys=keys[1], coeff_range=(0.0, 0.05), prob=0.7),
                RandGibbsNoised(keys=keys[1], alpha=(0.2, 0.4), prob=0.7),
            ], weights=[1.0, 1.0, 1.0]),
        ])
        # Simulate different acquisitions
        transforms.extend([
            OneOf(transforms=[
                RandAdjustContrastd(keys=keys[0], gamma=(0.9, 1.1), prob=1.0),
                RandSimulateLowResolutiond(keys=keys[0], prob=0.5, zoom_range=(0.8, 1.0)),
            ], weights=[1.0, 1.0]),
            OneOf(transforms=[
                RandAdjustContrastd(keys=keys[1], gamma=(0.9, 1.1), prob=1.0),
                RandSimulateLowResolutiond(keys=keys[1], prob=0.5, zoom_range=(0.8, 1.0)),
            ], weights=[1.0, 1.0]),
        ])
    
    # Get all keys required for padding and cropping for volume(s) 
    # and [optionally] target(s)
    if recon:
        keys_with_recon_1 = [keys[0], f"{keys[0]}_recon"]
        keys_with_recon_2 = [keys[1], f"{keys[1]}_recon"]
    else:
        keys_with_recon_1 = keys[0]
        keys_with_recon_2 = keys[1]
    keys_with_recon = [*keys_with_recon_1, *keys_with_recon_2]
    
    # Pad and crop to match input size
    transforms.extend([
        SpatialPadd(keys=keys_with_recon, spatial_size=input_size),
    ])
    if not val_mode:
        if conservative_mode:
            transforms.extend([
                RandSpatialCropd(keys=keys_with_recon, roi_size=input_size),
            ])
        else:
            transforms.extend([
                RandSpatialCropd(keys=keys_with_recon_1, roi_size=input_size),
                RandSpatialCropd(keys=keys_with_recon_2, roi_size=input_size),
            ])
    else:
        transforms.extend([
            CenterSpatialCropd(keys=keys_with_recon, roi_size=input_size),
        ])
    return cast(Callable[[Dict[str, NDArray]], Dict[str, Tensor]], Compose(transforms))