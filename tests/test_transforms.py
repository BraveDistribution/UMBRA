"""Tests and visualizations for transforms."""

from typing import Sequence
from typing import cast

import pytest
import numpy as np

from transforms import (
    get_mae_transforms,
    get_contrastive_transforms,
)


def generate_random_4d_volume(shape):
    """Generate a random 4D numpy array with channel dimension first."""
    return np.random.rand(*shape)


class TestTransformShapes:
    """Group of tests for ensuring deterministic output shapes"""
    @pytest.mark.parametrize("input_size", [
        (1, 72, 72, 72),
        (1, 96, 96, 96),
        (1, 96, 96, 88),
        (1, 120, 100, 102),
    ])
    def test_mae_transforms_equalize_sizes(self, input_size):
        """
        Test that MAE transforms ensure equal output sizes for 
        differently-sized inputs.
        """
        transforms = get_mae_transforms(
            keys=("volume",),
            input_size=(96, 96, 96),
            val_mode=False 
        )
        result = transforms(
            {"volume": generate_random_4d_volume(input_size)}
        )
        assert result["volume"].shape == (1, 96, 96, 96)
        assert result["volume_recon"].shape == (1, 96, 96, 96)

    @pytest.mark.parametrize(["vol_1", "vol_2"], [
        ((1, 72, 72, 72), (1, 96, 96, 96)),
        ((1, 96, 96, 96), (1, 96, 96, 88)),
        ((1, 96, 96, 88), (1, 120, 100, 102)),
        ((1, 120, 100, 102), (1, 72, 72, 72)),
    ],)
    def test_contrastive_transforms_equalize_sizes(self, vol_1, vol_2):
        """
        Test that contrastive transforms ensure equal output sizes 
        for differently-sized inputs.
        """
        transforms = get_contrastive_transforms(
            keys=("vol1", "vol2"),
            input_size=(96, 96, 96),
            val_mode=False,
            recon=True,
        )
        result = transforms({
            "vol1": generate_random_4d_volume(vol_1),
            "vol2": generate_random_4d_volume(vol_2),
        })
        assert result["vol1"].shape == (1, 96, 96, 96)
        assert result["vol2"].shape == (1, 96, 96, 96)
        assert result["vol1_recon"].shape == (1, 96, 96, 96)
        assert result["vol2_recon"].shape == (1, 96, 96, 96)