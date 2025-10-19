"""Tests and visualizations for transforms."""

from typing import Dict
from typing import cast

import pytest
import numpy as np
# pyright: reportPrivateImportUsage=false
from monai.transforms import (
    Compose,
    ToNumpyd,
)

from transforms import (
    get_mae_transforms,
    get_contrastive_transforms,
)   
from utils.visualization import plot_npy_volumes
from utils.io import load_volume


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

@pytest.mark.viz
class TestVisualizeTransforms:
    """Group of tests for visualizing transforms."""
    @pytest.fixture
    def volume(self):
        """Load example volume."""
        try:
            volume = load_volume("tests/examples/t1.npy")
        except FileNotFoundError:
            volume = generate_random_4d_volume((1, 110, 103, 112))
        return volume

    def test_mae_train_transforms(self, volume):
        """Visualize MAE training transforms."""
        transforms: Compose = get_mae_transforms(
            keys=("volume",),
            input_size=(96, 96, 96),
            val_mode=False,
        ).set_random_state(42)

        convert_to_numpy = ToNumpyd(keys=("volume", "volume_recon"))

        result = cast(Dict, transforms({"volume": volume}))
        result = cast(Dict[str, np.ndarray], convert_to_numpy(result))
        
        plot_npy_volumes({
            "original": volume, 
            "transformed": result["volume"], 
            "recon": result["volume_recon"]
        },
        title="MAE Training Transforms"
        )
        assert True
    
    def test_mae_val_transforms(self, volume):
        """Visualize MAE validation transforms."""
        transforms: Compose = get_mae_transforms(
            keys=("volume",),
            input_size=(96, 96, 96),
            val_mode=True,
        ).set_random_state(30)

        convert_to_numpy = ToNumpyd(keys=("volume", "volume_recon"))

        result = cast(Dict, transforms({"volume": volume}))
        result = cast(Dict[str, np.ndarray], convert_to_numpy(result))
        plot_npy_volumes({
            "original": volume, 
            "transformed": result["volume"], 
            "recon": result["volume_recon"]
        },
        title="MAE Validation Transforms"
        )
        assert True
    
    def test_contrastive_train_transforms(self, volume):
        """Visualize contrastive training transforms."""
        transforms: Compose = get_contrastive_transforms(
            keys=("vol1", "vol2"),
            input_size=(96, 96, 96),
            conservative_mode=True,
            val_mode=False,
            recon=True,
        ).set_random_state(42)

        convert_to_numpy = ToNumpyd(keys=("vol1", "vol2", "vol1_recon", "vol2_recon"))

        result = cast(Dict, transforms({"vol1": volume, "vol2": volume}))
        result = cast(Dict[str, np.ndarray], convert_to_numpy(result))
        plot_npy_volumes({
            "original": volume, 
            "vol1": result["vol1"], 
            "vol2": result["vol2"], 
            "vol1_recon": result["vol1_recon"], 
            "vol2_recon": result["vol2_recon"]
        },
        title="Contrative (with Reconstruction) Training Transforms"
        )
        assert True
    
    def test_contrastive_val_transforms(self, volume):
        """Visualize contrastive validation transforms."""
        transforms: Compose = get_contrastive_transforms(
            keys=("vol1", "vol2"),
            input_size=(96, 96, 96),
            val_mode=True,
            recon=True,
        ).set_random_state(12)

        convert_to_numpy = ToNumpyd(keys=("vol1", "vol2", "vol1_recon", "vol2_recon"))

        result = cast(Dict, transforms({"vol1": volume, "vol2": volume}))
        result = cast(Dict[str, np.ndarray], convert_to_numpy(result))
        plot_npy_volumes({
            "original": volume, 
            "vol1": result["vol1"], 
            "vol2": result["vol2"], 
            "vol1_recon": result["vol1_recon"], 
            "vol2_recon": result["vol2_recon"]
        },
        title="Contrative (with Reconstruction) Validation Transforms"
        )
        assert True