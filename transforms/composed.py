"""
MONAI-compatible composed transforms.
"""

from __future__ import annotations
from typing import Sequence

# pyright: reportPrivateImportUsage=false
from monai.transforms import (
    Compose,
)

from transforms.unit import (
    GetReconstructionTargetd, 
    CreateRandomMaskd,
)


def get_mae_transforms(
    mask_ratio: float | Sequence[float] = 0.6,
    mask_patch_size: int = 4,
) -> Compose:
    """
    Get MAE transforms.
    """
    return Compose([
        GetReconstructionTargetd(keys="img", recon_key="recon"),
    ])