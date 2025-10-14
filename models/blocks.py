"""
Pytorch blocks of layers to be used in the construction of networks.
"""
from __future__ import annotations

from typing import Any, Sequence, Optional, Dict, Literal
from typing import TYPE_CHECKING

import torch.nn as nn
import torch.nn.functional as F
from monai.networks.nets.swin_unetr import SwinTransformer

from utils.misc import ensure_tuple_dim
from utils.spatial import upsample_to, voxel_shuffle_3d

if TYPE_CHECKING:
    from torch import Tensor

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
        
    def forward(self, x: Tensor) -> Tensor:
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        return x


class SwinEncoder(SwinTransformer):
    """
    Wrapper for MONAI's `SwinTransformer`.
    """
    def __init__(
        self, 
        in_channels: int = 1,
        patch_size: int | Sequence[int] = 2,
        depths: Sequence[int] = (2, 2, 6, 2),
        num_heads: Sequence[int] = (3, 6, 12, 24),
        window_size: Sequence[int] | int = 7,
        feature_size: int = 48,
        use_v2: bool = True,
        spatial_dims: int = 3,
        **kwargs,
    ):
        super().__init__(
            in_chans=in_channels,
            embed_dim=feature_size,
            depths=depths,
            num_heads=num_heads,
            window_size=ensure_tuple_dim(window_size, spatial_dims),
            patch_size=ensure_tuple_dim(patch_size, spatial_dims),
            use_v2=use_v2,
            **kwargs,
        )


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

    def forward(self, feats: list[Tensor]) -> Tensor:
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
        p4 = self.lat4(f4) + upsample_to(p5, f4)
        p4 = self.smooth4(p4)

        p3 = self.lat3(f3) + upsample_to(p4, f3)
        p3 = self.smooth3(p3)

        p2 = self.lat2(f2) + upsample_to(p3, f2)
        p2 = self.smooth2(p2)

        p1 = self.lat1(f1) + upsample_to(p2, f1)
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

    def forward(self, x: Tensor) -> Tensor:
        return voxel_shuffle_3d(self.proj(x), self.up)