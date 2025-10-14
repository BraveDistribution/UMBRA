"""
Pytorch blocks of layers to be used in the construction of networks.
"""
from __future__ import annotations

from typing import Any, Sequence, Optional, Dict, Literal, Optional
from typing import TYPE_CHECKING
from contextlib import contextmanager

import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.networks.blocks.patchembedding import PatchEmbed

from utils.misc import ensure_tuple_dim
from utils.spatial import upsample_to_3d, voxel_shuffle_3d


class ConvNormAct3d(nn.Module):
    """
    Convolutional block with normalization and activation applied
    in the order: conv -> norm -> act.
    """
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        k: int = 3,
        s: int = 1,
        p: int = 1,
        norm: Literal["instance", "group", "batch"] = "instance",
        act: Literal["relu", "gelu", "leaky_relu"] = "gelu",
        norm_kwargs: Optional[Dict[str, Any]] = None,
        act_kwargs: Optional[Dict[str, Any]] = None,
    ):
        """
        Args:
            in_ch: Number of input channels
            out_ch: Number of output channels
            k: Kernel size
            s: Stride
            p: Padding
            norm: Normalization layer
            act: Activation function
            norm_kwargs: Keyword arguments for the normalization layer
            act_kwargs: Keyword arguments for the activation function
        """
        super().__init__()
        # Conv
        self.conv = nn.Conv3d(in_ch, out_ch, kernel_size=k, stride=s, padding=p, bias=False)
        
        # Norm
        norm_kwargs = norm_kwargs or {}
        if norm == "instance":
            self.norm = nn.InstanceNorm3d(out_ch, **norm_kwargs)
        elif norm == "group":
            n = norm_kwargs.pop("num_groups", 8)
            self.norm = nn.GroupNorm(num_groups=min(n, out_ch), num_channels=out_ch, **norm_kwargs)
        elif norm == "batch":
            self.norm = nn.BatchNorm3d(out_ch, **norm_kwargs)
        else:
            raise ValueError(f"Invalid normalization layer: {norm}")

        # Act
        act_kwargs = act_kwargs or {}
        if act == "relu":
            self.act = nn.ReLU(**act_kwargs)
        elif act == "gelu":
            self.act = nn.GELU(**act_kwargs)
        elif act == "leaky_relu":
            self.act = nn.LeakyReLU(**act_kwargs)
        else:
            raise ValueError(f"Invalid activation function: {act}")
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        return x


class FPNDecoderFeaturesOnly(nn.Module):
    """
    FPN that fuses 5 encoder scales (high->low: f1..f5) into a narrow feature map 
    at f1 resolution.
    """
    def __init__(
        self,
        in_feats: Sequence[int],
        in_chans: int,
        out_channels: int,
        width: int = 32,
        norm: Literal["instance", "group", "batch"] = "instance",
        norm_kwargs: Optional[Dict[str, Any]] = None,
        use_smooth: bool = True,  
    ):
        super().__init__()
        assert len(in_feats) == 5, "Need 5 scales [f1..f5]"
        f1, f2, f3, f4, f5 = in_feats

        # Laterals (1x1x1) to common width
        self.lat1 = nn.Conv3d(f1, width, kernel_size=1, bias=False)
        self.lat2 = nn.Conv3d(f2, width, kernel_size=1, bias=False)
        self.lat3 = nn.Conv3d(f3, width, kernel_size=1, bias=False)
        self.lat4 = nn.Conv3d(f4, width, kernel_size=1, bias=False)
        self.lat5 = nn.Conv3d(f5, width, kernel_size=1, bias=False)

        # Optional light 3x3x3 smoothing after top-down adds
        if use_smooth:
            self.smooth4 = ConvNormAct3d(width, width, k=3, p=1, norm=norm, norm_kwargs=norm_kwargs)
            self.smooth3 = ConvNormAct3d(width, width, k=3, p=1, norm=norm, norm_kwargs=norm_kwargs)
            self.smooth2 = ConvNormAct3d(width, width, k=3, p=1, norm=norm, norm_kwargs=norm_kwargs)
            self.smooth1 = ConvNormAct3d(width, width, k=3, p=1, norm=norm, norm_kwargs=norm_kwargs)
        else:
            self.smooth4 = self.smooth3 = self.smooth2 = self.smooth1 = nn.Identity()

    def forward(self, feats: list[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            feats: list of 5 tensors [f1,f2,f3,f4,f5] with shapes
                   f1: (B,C1,D1,H1,W1) ... f5: (B,C5,D5,H5,W5), where
                   D1,H1,W1 are the finest token grid (typically image/P).
        Returns:
            p1: fused features at f1 resolution, shape (B, width, D1, H1, W1)
        """
        if len(feats) != 5:
            raise ValueError(f"[{self.__class__.__name__}] Expected 5 features, got {len(feats)}.")
        f1, f2, f3, f4, f5 = feats

        p5 = self.lat5(f5)
        p4 = self.lat4(f4) + upsample_to_3d(p5, f4)
        p4 = self.smooth4(p4)

        p3 = self.lat3(f3) + upsample_to_3d(p4, f3)
        p3 = self.smooth3(p3)

        p2 = self.lat2(f2) + upsample_to_3d(p3, f2)
        p2 = self.smooth2(p2)

        p1 = self.lat1(f1) + upsample_to_3d(p2, f1)
        p1 = self.smooth1(p1)

        return p1


class VoxelShuffleHead3D(nn.Module):
    """
    Voxel shuffle head for 3D tensors.

    Applies a 1x1x1 convolution to the get target number of channels and
    then gets original spatial dimensions back via voxel shuffle.
    """
    def __init__(self, in_ch: int, out_ch: int, up: Sequence[int] | int):
        """
        Args:
            in_ch: Number of input channels
            out_ch: Number of output channels
            up: Upscale factor; set equal to ViT patch size
        """
        super().__init__()
        self.up = ensure_tuple_dim(up, 3)
        target_dims = out_ch * self.up[0] * self.up[1] * self.up[2]
        self.proj = nn.Conv3d(in_ch, target_dims, kernel_size=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return voxel_shuffle_3d(self.proj(x), self.up)


class MaskTokenInjector(nn.Module):
    """
    Mask patches and replace with a learnable mask token.
    """
    def __init__(self, embed_dim: int):
        """
        Args:
            embed_dim: Mask token dimension; should match ViT embedding dimension.
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        nn.init.trunc_normal_(self.mask_token, std=0.02)
    
    def forward(self, tokens: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            tokens: (B, N, C) from patch embed
            mask: (B, N) boolean; True => masked patch
        """
        B, N, C = tokens.shape
        assert mask.shape == (B, N), f"Expected mask shape {B}x{N}, got {mask.shape}"
        assert C == self.embed_dim, f"Expected {self.embed_dim} channels, got {C}"

        mask_tokens = self.mask_token.expand(B, N, C)
        return torch.where(mask.unsqueeze(-1), mask_tokens, tokens)


class PatchEmbedWithMask(nn.Module):
    """
    Wrapper for MONAI's `PatchEmbed` that injects mask tokens.
    """
    def __init__(self, patch_embed: PatchEmbed, embed_dim: int):
        super().__init__()
        self.patch_embed = patch_embed
        self.mask_token_injector = MaskTokenInjector(embed_dim)
        self._mask: Optional[torch.Tensor] = None  # (B, N) bool, will get populated in each call

    def set_mask(self, mask_or_none):
        self._mask = mask_or_none

    @contextmanager
    def use_mask(self, mask_or_none):
        """
        Context manager to set the mask for the duration of the MAE forward
        pass and restore it in case of exception.

        Same encoder API when in contrastive mode or inference by setting/leaving
        it to None. Leaves encoder's `forward()` unchanged.

        Example:
        ```python
        >>> with model.patch_embed.use_mask(mask):
        ...    feats = model(x)
        ```
        
        """
        old = self._mask
        self._mask = mask_or_none
        try:
            yield
        finally:
            self._mask = old

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (B, C, *spatial_dims)
            mask: Boolean tensor of shape (B, N)
        """
        # MONAI PatchEmbed -> grid (B, C, Dp, Hp, Wp) or (B, C, Hp, Wp)
        grid = self.patch_embed(x)
        if self._mask is None:
            return grid
        
        # flatten grid -> (B, N, C)
        # 3D case: (B, C, Dp, Hp, Wp)
        if grid.dim() == 5:
            B, C, Dp, Hp, Wp = grid.shape
            tokens = grid.permute(0, 2, 3, 4, 1).contiguous().view(B, Dp*Hp*Wp, C)
            if self._mask.dim() == 4:  # (B,Dp,Hp,Wp) support
                mask = self._mask.view(B, Dp*Hp*Wp)
            else:
                mask = self._mask
            tokens = self.mask_token_injector(tokens, mask)
            # reshape back to grid
            grid = tokens.view(B, Dp, Hp, Wp, C).permute(0, 4, 1, 2, 3).contiguous()
        
        # 2D case: (B, C, Hp, Wp)
        elif grid.dim() == 4:
            B, C, Hp, Wp = grid.shape
            tokens = grid.permute(0, 2, 3, 1).contiguous().view(B, Hp*Wp, C)
            if self._mask.dim() == 3:
                mask = self._mask.view(B, Hp*Wp)
            else:
                mask = self._mask
            tokens = self.mask_token_injector(tokens, mask)
            grid = tokens.view(B, Hp, Wp, C).permute(0, 3, 1, 2).contiguous()
        else:
            raise RuntimeError("Unexpected PatchEmbed output rank.")
        return grid

        
