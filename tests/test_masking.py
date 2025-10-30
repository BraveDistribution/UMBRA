"""Unit and integration tests for all masking operations in MAE pretraining."""

from __future__ import annotations

from typing import cast, Dict

import pytest
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from monai.transforms.utility.dictionary import ToNumpyd
from monai.networks.blocks.patchembedding import PatchEmbed
from monai.transforms.compose import Compose

from utils.masking import generate_random_mask_conv, up_to_voxel_space
from utils.visualization import plot_npy_volumes
from utils.data  import load_volume
from models.blocks import (
    MaskTokenInjector,
    PatchEmbedWithMask,
)
from models.networks import (
    SwinEncoder,
    SwinEncoderMAE,
    SwinMAE,
)
from transforms.composed import get_mae_transforms


def generate_random_4d_volume(shape):
    """Generate a random 4D numpy array with channel dimension first."""
    return np.random.rand(*shape)

def _ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b

def _downsample(inp: torch.Tensor) -> torch.Tensor:
    """Downsamples a 3D tensor by a factor of 2 using nearest-neighbor interpolation."""
    inp = inp.float()
    inp = inp.unsqueeze(1)  # shape: (B, 1, D, H, W)
    inp_ds = F.interpolate(inp, scale_factor=0.5, mode="nearest")
    return inp_ds.squeeze(1)  # (B, D, H, W)

@pytest.fixture(autouse=True)
def mock_patch_embed(monkeypatch):
    """Mock the `PatchEmbed` block to return a fixed shape tensor."""
    def dummy_forward(self, x: torch.Tensor) -> torch.Tensor:
        gen = torch.Generator().manual_seed(42)
        return torch.randn(1, 24, 48, 48, 48, generator=gen)
    monkeypatch.setattr(PatchEmbed, "forward", dummy_forward)

@pytest.fixture(autouse=True)
def mock_swin_encoder(monkeypatch):
    """Mock the `SwinEncoder` block to return fixed shape feature maps."""
    def dummy_forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        gen = torch.Generator().manual_seed(42)
        return [
            torch.randn(1, 48, 48, 48, 48, generator=gen),
            torch.randn(1, 96, 24, 24, 24, generator=gen),
            torch.randn(1, 192, 12, 12, 12, generator=gen),
            torch.randn(1, 384, 6, 6, 6, generator=gen),
            torch.randn(1, 768, 3, 3, 3, generator=gen),
        ]
    monkeypatch.setattr(SwinEncoder, "forward", dummy_forward)


class TestMaskGeneration: 
    """Tests for mask generation."""
    def test_boolean_mask(self):
        """Test that the mask is a boolean mask."""
        mask = generate_random_mask_conv(
            input_size=(96, 96, 96),
            patch_size=(2, 2, 2),
            num_downsamples=4,
            batch_size=1,
            mask_ratio=0.5,
            return_kind="flat",
        )
        assert mask.dtype == torch.bool
        
    @pytest.mark.parametrize("input_size", [
        (1, 1, 1),
        (96, 96, 96),
        (96, 96, 88),
        (120, 100, 102),
    ])
    def test_generate_random_mask_flat(self, input_size):
        """Test that generate_random_mask_flat generates a flat mask."""
        mask = generate_random_mask_conv(
            input_size=input_size,
            patch_size=(2, 2, 2),
            num_downsamples=4,
            batch_size=1,
            mask_ratio=(0.5, 0.75),
            return_kind="flat",
        )
        n_patches_after_patch_embed = (
            _ceil_div(input_size[0], 2) * _ceil_div(input_size[1], 2) * _ceil_div(input_size[2], 2)
        )
        assert mask.shape == (1, n_patches_after_patch_embed)

    @pytest.mark.parametrize("input_size", [
        (1, 1, 1),
        (96, 96, 96),
        (96, 96, 88),
        (120, 100, 102),
    ])
    def test_generate_random_mask_grid(self, input_size):
        """Test that generate_random_mask_grid generates a grid mask."""
        mask = generate_random_mask_conv(
            input_size=input_size,
            patch_size=(2, 2, 2),
            num_downsamples=4,
            batch_size=1,
            mask_ratio=(0.5, 0.75),
            return_kind="grid",
        )
        assert mask.shape == (
            1, _ceil_div(input_size[0], 2), _ceil_div(input_size[1], 2), _ceil_div(input_size[2], 2)
        )
    
    @pytest.mark.parametrize("input_size", [
        (1, 1, 1),
        (96, 96, 96),
        (96, 96, 88),
        (120, 100, 102),
    ])
    def test_generate_random_mask_voxel(self, input_size):
        """Test that generate_random_mask_voxel generates a voxel mask."""
        torch.manual_seed(42)
        mask = generate_random_mask_conv(
            input_size=input_size,
            patch_size=(2, 2, 2),
            num_downsamples=4,
            batch_size=1,
            mask_ratio=(0.5, 0.75),
            return_kind="voxel",
        )
        assert mask.shape == (1, 1, input_size[0], input_size[1], input_size[2])

        # Test equivalence to `grid` -> `voxel` (will be the case in MAE training)
        torch.manual_seed(42)
        mask_grid = generate_random_mask_conv(
            input_size=input_size,
            patch_size=(2, 2, 2),
            num_downsamples=4,
            batch_size=1,
            mask_ratio=(0.5, 0.75),
            return_kind="grid",
        )
        mask_voxel = up_to_voxel_space(mask_grid, input_size, (2, 2, 2))
        assert torch.allclose(mask, mask_voxel)

    @pytest.mark.parametrize("input_size", [
        (96, 96, 96),
        (96, 96, 88),
        (120, 100, 102),
    ])
    def test_bottleneck_masked(self, input_size):
        """Test that the bottleneck is masked -> essential to avoid leakage"""
        mask = generate_random_mask_conv(
            input_size=input_size,
            patch_size=(2, 2, 2),
            num_downsamples=4,
            batch_size=1,
            mask_ratio=(0.5, 0.75),
            return_kind="grid",
        )
        for _ in range(4):
            mask = _downsample(mask)
        mask_flat = mask.view(mask.size(0), -1)

        unique_vals = torch.unique(mask_flat)
        assert set(unique_vals.tolist()).issubset({0.0, 1.0}), f"Found values: {unique_vals}"

    @pytest.mark.parametrize("mask_ratio", [0.0, 0.25, 0.5, 0.75, 1.0])
    def test_fixed_mask_ratio(self, mask_ratio):
        """Test that the mask ratio is correct."""
        mask = generate_random_mask_conv(
            input_size=(96, 96, 96),
            patch_size=(2, 2, 2),
            num_downsamples=4,
            batch_size=1,
            mask_ratio=mask_ratio,
            return_kind="flat",
        )
        assert torch.allclose(torch.mean(mask.float()), torch.tensor(mask_ratio), atol=0.02)

    def test_variable_mask_ratio(self):
        """Test that the mask ratio is varying within the range."""
        for _ in range(10):
            mask = generate_random_mask_conv(
                input_size=(32, 32, 32),
                patch_size=(2, 2, 2),
                num_downsamples=2,
                batch_size=1,
                mask_ratio=(0.6, 0.7),
                return_kind="flat",
            )
            assert 0.58 <= torch.mean(mask.float()) <= 0.72
    
    @pytest.mark.viz
    def test_visualize_upsampled_mask(self):
        """Visualize the upsampled mask."""
        try:
            volume = load_volume("tests/examples/t1.npy")
        except FileNotFoundError:
            volume = generate_random_4d_volume((1, 110, 103, 112))

        transforms: Compose = get_mae_transforms(
            keys=("volume",),
            input_size=(96, 96, 96),
            val_mode=False,
        ).set_random_state(22)

        transformed = cast(Dict, transforms({"volume": volume}))

        mask = generate_random_mask_conv(
            input_size=transformed["volume"].shape[-3:],
            patch_size=(2, 2, 2),
            num_downsamples=4,
            batch_size=1,
            mask_ratio=(0.5, 0.75),
            return_kind="grid",
        )
        mask_voxel = up_to_voxel_space(
            mask, transformed["volume"].shape[-3:], (2, 2, 2)
        )
        dict_data: Dict = {
            "original": transformed["volume"],
            "mask_voxel": mask_voxel.squeeze(0),
            "masked_volume": transformed["volume"] * ~mask_voxel.squeeze(0),
            "recon": transformed["volume_recon"],
        }
        dict_data = ToNumpyd(keys=list(dict_data.keys()))(dict_data)
        plot_npy_volumes(dict_data)
        assert True


class TestMaskInjectionBlock:
    """Tests proper injection of mask token to input embeddings."""
    @pytest.fixture
    def mask(self):
        """Generate a random mask."""
        return generate_random_mask_conv(
            input_size=(96, 96, 96),
            patch_size=(2, 2, 2),
            num_downsamples=4,
            batch_size=1,
            mask_ratio=(0.5, 0.75),
            return_kind="flat",
        )
    
    @pytest.fixture
    def input_embed(self):
        """Generate random input embeddings."""
        return torch.randn(1, 110592, 24)
    
    def test_mask_injection(self, mask, input_embed):
        """Test that the mask is injected correctly."""
        mask_injector = MaskTokenInjector(embed_dim=24)
        masked_embed = mask_injector(input_embed, mask)
        assert masked_embed.shape == input_embed.shape
        assert torch.allclose(masked_embed[~mask], input_embed[~mask])
        assert not torch.allclose(masked_embed[mask], input_embed[mask])
    
    @pytest.mark.parametrize("embed_dim", [0, 12, 36, 48])
    def test_wrong_embed_dim(self, mask, input_embed, embed_dim):
        """
        Test that the mask injector raises an error if the patch embedding
        dimension does not match the specified mask token dimension.
        """
        mask_injector = MaskTokenInjector(embed_dim=embed_dim)
        with pytest.raises(AssertionError):
            mask_injector(input_embed, mask)    
    
    def test_wrong_mask_shape(self, input_embed):
        """
        Test that the mask injector raises an error if the mask shape does not
        match expected shape: (B, N) where N is the number of patches.
        """
        masks = [
            torch.randn(1, 110592, 2).bool(), # extra dim
            torch.randn(1, 110592, 2, 2).bool(), # extra dim
            torch.randn(110592).bool(), # missing dim
            torch.randn(1, 60296).bool(), # wrong number of patches
        ]
        mask_injector = MaskTokenInjector(embed_dim=24)
        for mask in masks:
            with pytest.raises(AssertionError):
                mask_injector(input_embed, mask)

        # test that it works with the correct shape
        mask_injector(input_embed, torch.randn(1, 110592).bool())
        assert True

    def test_wrong_input_type(self, input_embed):
        """
        Test that the mask injector raises an error if the input embeddings
        are not a tensor of bool type.
        """
        mask_injector = MaskTokenInjector(embed_dim=24)
        types = [torch.float, torch.int, torch.long, torch.double]
        for type in types:
            with pytest.raises(RuntimeError):
                mask_injector(input_embed, torch.randn(1, 110592).type(type))


class TestPatchEmbedWithMask:
    """
    Tests proper behavior of patch embedding with masking, as a drop-in 
    replacement for the original `PatchEmbed` block.
    """
    @pytest.fixture
    def op(self):
        """Generate a patch embed with mask block."""
        return PatchEmbedWithMask(
            patch_embed=PatchEmbed(embed_dim=24),
            embed_dim=24,
        )

    @pytest.fixture
    def mask(self):
        """Generate a random mask."""
        torch.manual_seed(42)
        return generate_random_mask_conv(
            input_size=(96, 96, 96),
            patch_size=(2, 2, 2),
            num_downsamples=4,
            batch_size=1,
            mask_ratio=(0.5, 0.75),
            return_kind="grid",
        )
    
    def test_no_op(self, op):
        """
        Test that the forward pass works as the original `PatchEmbed` block
        when no mask is provided.
        """
        x = torch.randn(1, 1, 48, 48, 48)
        assert torch.allclose(op(x), PatchEmbed(embed_dim=24)(x))

    def test_masking(self, op, mask):
        """
        Test that the forward pass works as expected when a mask is provided.
        """
        x = torch.randn(1, 1, 96, 96, 96)
        with op.use_mask(mask):
            out = op(x)
        assert out.shape == (1, 24, 48, 48, 48)
       
        # Get reference using `PatchEmbed` + `MaskTokenInjector`
        patch_embed = PatchEmbed(embed_dim=24)(x) # (B, C, Dp, Hp, Wp)
        flattened_embed = patch_embed.permute(0, 2, 3, 4, 1).contiguous().view(1, 48*48*48, 24) # (B, N, C)
        flattened_mask = mask.view(1, 48*48*48) # (B, N)
        masked_embed = op.mask_token_injector(flattened_embed, flattened_mask) # (B, N, C)
        reshaped_embed = masked_embed.view(1, 48, 48, 48, 24).permute(0, 4, 1, 2, 3).contiguous() # (B, C, Dp, Hp, Wp)
        assert torch.allclose(out, reshaped_embed)

    def test_handles_grid_and_flat_masks(self, op, mask):
        """
        Test support for both grid (B, Dp, Hp, Wp) and flat (B, Dp*Hp*Wp) masks.
        """
        x = torch.randn(1, 1, 96, 96, 96)
        with op.use_mask(mask): # `mask` fixture gives a grid mask
            out_grid = op(x)
        
        torch.manual_seed(42)
        flat_mask = generate_random_mask_conv(
            input_size=(96, 96, 96),
            patch_size=(2, 2, 2),
            num_downsamples=4,
            batch_size=1,
            mask_ratio=(0.5, 0.75),
            return_kind="flat",
        )
        with op.use_mask(flat_mask):
            out_flat = op(x)
        
        assert torch.allclose(out_grid, out_flat)
    
    def test_mask_setting_api(self, op, mask):
        """
        Test the `set_mask` and `use_mask` APIs.
        """
        x = torch.randn(1, 1, 96, 96, 96)
        out_no_mask = op(x)

        with op.use_mask(mask):
            out_mask = op(x)

        op.set_mask(mask)
        out_mask_setter = op(x)

        with op.use_mask(None):
            out_mask_forced_none = op(x)

        assert torch.allclose(out_mask, out_mask_setter)
        assert torch.allclose(out_no_mask, out_mask_forced_none)
        assert not torch.allclose(out_mask, out_no_mask)
    
    def test_raises_error_on_unexpected_rank(self, op):
        """
        Test that the forward pass raises an error if the input tensor
        has an unexpected rank.
        """
        wrong_tensors = [
            torch.randn(1, 1, 96, 96, 96, 96), # 6D
            torch.randn(1, 1, 96), # 3D
            torch.randn(1, 1), # 2D
            torch.randn(1), # 1D
        ]
        for x in wrong_tensors:
            with pytest.raises(RuntimeError):
                op(x)
    

class TestSwinEncoderMAE:
    """Tests proper initialization of `SwinEncoderMAE`."""
    @pytest.fixture
    def model(self):
        return SwinEncoderMAE()

    def test_patch_embed_replacement(self, model):
        """Test that the `PatchEmbed` block is replaced with `PatchEmbedWithMask`."""
        assert isinstance(model.patch_embed, PatchEmbedWithMask)

    def test_no_instance_or_batch_norm(self, model):
        """Test that no `InstanceNorm` or `BatchNorm` layers are present."""
        for name, child in model.named_children():
            if isinstance(child, (nn.InstanceNorm2d, nn.InstanceNorm3d, nn.BatchNorm2d, nn.BatchNorm3d)):
                assert False, f"Found {name} with {type(child)}"
        assert True


class TestSwinMAE:
    """Tests masking and decoding support in `SwinMAE`."""
    @pytest.fixture
    def model(self):
        return SwinMAE()

    def test_masking_and_decoding(self, model):
        """Test that the masking and decoding support works as expected."""
        x = torch.randn(1, 1, 96, 96, 96)
        mask = generate_random_mask_conv(
            input_size=(96, 96, 96),
            patch_size=(2, 2, 2),
            num_downsamples=4,
            batch_size=1,
            mask_ratio=(0.5, 0.75),
            return_kind="grid",
        )
        out = model(x, mask)
        assert out.shape == (1, 1, 96, 96, 96)