"""
MONAI-compatible composed transforms.
"""

from __future__ import annotations

__all__ = [
    "get_mae_transforms",
    "get_contrastive_transforms",
    "get_segmentation_transforms",
    "load_nifti",
]

from typing import Sequence, Union, Literal, Optional

# pyright: reportPrivateImportUsage=false
from monai.transforms import (
    CenterSpatialCropd,
    Compose,
    ConcatItemsd,
    DeleteItemsd,
    EnsureChannelFirstd,
    LoadImaged,
    Orientationd,
    NormalizeIntensityd,
    OneOf,
    RandFlipd,
    RandAffined,
    RandAdjustContrastd,
    RandBiasFieldd,
    RandCropByPosNegLabeld,
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
    PadToMaxOfKeysd,
    ClipNonzeroPercentilesd,
)

def get_mae_transforms(
    keys: Sequence[str] = ("volume",),
    input_size: Union[int, Sequence[int]] = 96,
    val_mode: bool = False,
) -> Compose:
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
    # Standardize inputs
    transforms = [
        ToTensord(keys=keys, track_meta=False),
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
        SpatialPadd(keys=keys_with_recon, spatial_size=input_size, mode='edge'),
    ])
    if not val_mode:
        transforms.extend([
            RandSpatialCropd(keys=keys_with_recon, roi_size=input_size),
        ])
    else:
        transforms.extend([
            CenterSpatialCropd(keys=keys_with_recon, roi_size=input_size),
        ])

    return Compose(transforms)

def get_contrastive_transforms(
    keys: Sequence[str] = ("vol1", "vol2"),
    input_size: Union[int, Sequence[int]] = 96,
    conservative_mode: bool = True,
    val_mode: bool = False,
    recon: bool = False,
) -> Compose:
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
    # Standardize inputs
    transforms = [
        ToTensord(keys=keys, track_meta=False),
        EnsureChannelFirstd(keys=keys, channel_dim=0),
    ]

    # Spatial augmentations
    if not val_mode:
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

    # Ensure all keys have the same spatial size
    transforms.extend([
        PadToMaxOfKeysd(keys=keys, mode='edge'),
    ])
    
    # Get reconstruction targets
    if recon:
        transforms.extend([
            GetReconstructionTargetd(keys=keys[0], recon_key=f"{keys[0]}_recon"),
            GetReconstructionTargetd(keys=keys[1], recon_key=f"{keys[1]}_recon"),
        ])

    # Intensity augmentations
    if not val_mode:
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
        keys_with_recon_1 = [keys[0]]
        keys_with_recon_2 = [keys[1]]
    keys_with_recon = [*keys_with_recon_1, *keys_with_recon_2]
    
    # Pad and crop to match input size
    transforms.extend([
        SpatialPadd(keys=keys_with_recon, spatial_size=input_size, mode='edge'),
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
   
    return Compose(transforms)

def load_nifti(
    keys: Sequence[str] = ("volume",),
    mask_keys: Optional[Sequence[str]] = None,
    allow_missing_keys: bool = False,
    orientation: Literal['LPI', 'RAS', 'LAS'] = 'LPI',
) -> Compose:
    """
    Load NIfTI files and convert to numpy arrays.

    Args:
        keys: Keys to apply the transforms to.

    Returns:
        Compose object with the transforms.
    """
    all_keys = [*keys, *mask_keys] if mask_keys is not None else keys
    return Compose([
        LoadImaged(keys=all_keys, reader='NibabelReader', ensure_channel_first=True, 
                   allow_missing_keys=allow_missing_keys),
        Orientationd(keys=all_keys, axcodes=orientation, allow_missing_keys=allow_missing_keys),
        ClipNonzeroPercentilesd(keys=keys, lower=0.5, upper=99.5, allow_missing_keys=allow_missing_keys),
        NormalizeIntensityd(keys=keys, allow_missing_keys=allow_missing_keys, nonzero=True),
    ])

def get_segmentation_transforms(
    input_size: Union[int, Sequence[int]] = 96,
    keys: Sequence[str] = ("volume",),
    seg_key: str = "seg",
    out_key: str = "volume",
    n_patches: int = 4,
    n_pos: int = 1,
    n_neg: int = 2,
    val_mode: bool = False,
    allow_missing_keys: bool = True,
) -> Compose:
    """
    IO transforms + augmentations for segmentation tasks.

    Args:
        input_size: Input size.
        keys: Keys to apply transforms to.
        seg_key: Key with integer segmentation mask.
        out_key: Key to store the resulting volume, after concatenation across modalities.
        n_patches: Number of crops to extract per image.
        n_pos: Relative weight of positive crops (i.e., containing the mask).
        n_neg: Relative weight of negative crops (i.e., not containing the mask).
        val_mode: Whether to use for validation; no augmentations are applied.
        allow_missing_keys: Whether to allow missing keys.
    """
    # Standardize inputs
    transforms = [
        ToTensord(keys=keys, track_meta=False),
        EnsureChannelFirstd(keys=keys, channel_dim=0),
    ]

    # Padding and interpolation modes for images and segmentation mask
    all_keys = [*keys, seg_key]
    pad_mode_affine =  ['border'] * len(keys) + ['constant']
    pad_mode_spatial = ['edge'] * len(keys) + ['constant']
    interp_mode = ['bilinear'] * len(keys) + ['nearest']

    if not val_mode:
        # Spatial augmentations
        transforms.extend([
            RandFlipd(keys=all_keys, spatial_axis=0, prob=0.5, 
                    allow_missing_keys=allow_missing_keys),
            RandFlipd(keys=all_keys, spatial_axis=1, prob=0.5, 
                    allow_missing_keys=allow_missing_keys),
            RandAffined(keys=all_keys, rotate_range=(0.1, 0.1, 0.1),
                        scale_range=(0.1, 0.1, 0.1), mode=interp_mode, 
                        padding_mode=pad_mode_affine, prob=1.0,
                        allow_missing_keys=allow_missing_keys),
        ])
        # Intensity augmentations; unique for each modality
        for key in keys:
            transforms.extend([
                RandScaleIntensityFixedMeand(keys=key, factors=0.1, prob=0.8, 
                                            allow_missing_keys=allow_missing_keys),
                RandGaussianNoised(keys=key, std=0.01, prob=0.3, 
                                allow_missing_keys=allow_missing_keys),
            ])
            # Simulate artifacts
            transforms.extend([
                OneOf(transforms=[
                    RandGaussianSmoothd(keys=key, sigma_x=(0.5, 1.0), prob=0.7, 
                                        allow_missing_keys=allow_missing_keys),
                    RandBiasFieldd(keys=key, coeff_range=(0.0, 0.05), prob=0.7, 
                                allow_missing_keys=allow_missing_keys),
                    RandGibbsNoised(keys=key, alpha=(0.2, 0.4), prob=0.7, 
                                    allow_missing_keys=allow_missing_keys),
                ], weights=[1.0, 1.0, 1.0]),
            ])
            # Simulate different acquisitions
            transforms.extend([
                OneOf(transforms=[
                    RandAdjustContrastd(keys=key, gamma=(0.9, 1.1), prob=1.0, 
                                        allow_missing_keys=allow_missing_keys),
                    RandSimulateLowResolutiond(keys=key, prob=0.5, zoom_range=(0.8, 1.0),
                                            allow_missing_keys=allow_missing_keys),
                ], weights=[1.0, 1.0]),
            ])

    # Pad and crop to match input size
    transforms.extend([
        SpatialPadd(keys=all_keys, spatial_size=input_size, mode=pad_mode_spatial, 
                    allow_missing_keys=allow_missing_keys),
    ])
    if not val_mode:
        transforms.extend([
            RandCropByPosNegLabeld(keys=all_keys, label_key=seg_key, 
                                   spatial_size=input_size, 
                                   pos=n_pos, neg=n_neg, num_samples=n_patches,
                                   allow_missing_keys=allow_missing_keys),
        ])
    else:
        transforms.extend([
            CenterSpatialCropd(keys=all_keys, roi_size=input_size),
        ])
    
    # Concatenate across modalities
    transforms.extend([
        ConcatItemsd(keys=keys, name=out_key, dim=0,
                     allow_missing_keys=allow_missing_keys),
        DeleteItemsd(keys=keys)
    ])
   
    return Compose(transforms)