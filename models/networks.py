from typing import Any, Sequence, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

from monai.networks.nets.swin_unetr import SwinTransformer

from .blocks import (
    FPNDecoderFeaturesOnly, 
    VoxelShuffleHead3D,
    PatchEmbedWithMask,
)
from utils.misc import ensure_tuple_dim
from utils.nets import swap_in_to_gn


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


class SwinEncoderMAE(nn.Module):
    """
    Swin ViT encoder with mask token injection support for MAE.

    Same args as `SwinEncoder`.
    """
    def __init__(
        self,
        *,
        in_channels: int = 1,
        patch_size: int | Sequence[int] = 2,
        depths: Sequence[int] = (2, 2, 6, 2),
        num_heads: Sequence[int] = (3, 6, 12, 24),
        window_size: Sequence[int] | int = 7,
        feature_size: int = 48,
        use_v2: bool = True,
        spatial_dims: int = 3,
        **extra_swin_kwargs,
    ):
        super().__init__()
        self.encoder = SwinEncoder(
            in_channels=in_channels,
            patch_size=patch_size,
            depths=depths,
            num_heads=num_heads,
            window_size=window_size,
            feature_size=feature_size,
            use_v2=use_v2,
            spatial_dims=spatial_dims,
            **extra_swin_kwargs,
        )
        # Replace patch embed with patch embed + mask injector ->
        # -> allows masking on the token space
        patch_embed_wrapper = PatchEmbedWithMask(
            patch_embed=self.encoder.patch_embed, embed_dim=feature_size
        )
        self.encoder.patch_embed = patch_embed_wrapper # type: ignore[assignment]

        # Swap InstanceNorm to GroupNorm -> less sensitive to mask ratio
        swap_in_to_gn(self.encoder)


class Swinv2LateFusionFPNDecoder(nn.Module):
    """
    SwinViT v2 with lightweight FPN decoder.

    Supports multimodal input (concatenated in the channel dimension) via
    *late fusion* (sharing the encoder) or *early fusion* (sharing the decoder).
    """

    def __init__(
        self,
        *,
        # SwinViT encoder args
        in_channels: int = 1,
        patch_size: int | Sequence[int] = 2,
        depths: Sequence[int] = (2, 2, 6, 2),
        num_heads: Sequence[int] = (3, 6, 12, 24),
        window_size: Sequence[int] | int = 7,
        feature_size: int = 48,
        use_v2: bool = True,
        extra_swin_kwargs: dict[str, Any] | None = None,
        spatial_dims: int = 3,
        # Fusion args
        n_late_fusion: int = 1,
        # FPN decoder args
        out_channels: int = 1,
        width: int = 32,
        norm: Literal["instance", "group", "batch"] = "instance",
    ):
        super().__init__()

        extra_swin_kwargs = extra_swin_kwargs or {}

        self.encoder = SwinEncoder(
            in_channels=in_channels,
            feature_size=feature_size,
            depths=depths,
            num_heads=num_heads,
            window_size=window_size,
            patch_size=patch_size,
            use_v2=use_v2,
            **extra_swin_kwargs,
        )

        # MONAI's `SwinTransformer` returns input after patch_embed and 4-level feature maps
        in_feats = [feature_size * 2**i for i in range(len(depths) + 1)]
        self.L = len(in_feats)

        self.feats_decoder = FPNDecoderFeaturesOnly(
            in_feats=in_feats,
            in_chans=in_channels,
            out_channels=out_channels,
            width=width,
            norm=norm,
        )
        self.head = VoxelShuffleHead3D(
            in_ch=width,
            out_ch=out_channels,
            up=patch_size,
        )

        self.n_late_fusion = n_late_fusion
        self.in_channels = in_channels
        self.spatial_dims = spatial_dims
        self.fusion_weights = nn.Parameter(torch.zeros(1 + self.L, n_late_fusion))

    def _reshape_input(self, x: torch.Tensor) -> torch.Tensor:
        """
        (B, n_mod, *spatial) -> (B, n_mod, 1, *spatial)
        *or*
        (B, n_mod, C, *spatial) -> unchanged
        """
        if x.ndim == 2 + self.spatial_dims:  # (B, n_late_fusion, *spatial_dims)
            x = x.unsqueeze(2)  # (B, n_late_fusion, 1, *spatial_dims)
        elif x.ndim == 3 + self.spatial_dims:  # (B, n_late_fusion, C, *spatial_dims)
            pass
        else:
            msg = (
                f"[{self.__class__.__name__}] Expected input shape to be "
                f"(B, n_late_fusion, *spatial_dims) or (B, n_late_fusion, C, *spatial_dims), "
                f"but got {x.shape}."
            )
            raise ValueError(msg)
        return x

    def _encode_one(self, x1: torch.Tensor) -> list[torch.Tensor]:
        """Single encoder forward pass."""
        feats = self.encoder(x1)
        if not isinstance(feats, (list, tuple)):
            msg = f"[{self.__class__.__name__}] Encoder must return a list/tuple of multi-scale features."
            raise RuntimeError(msg)
        if len(feats) != self.L:
            msg = (
                f"[{self.__class__.__name__}] Expected {self.L} features from encoder, "
                f"got {len(feats)}."
            )
            raise RuntimeError(msg)
        return list(feats)  # [f1..fL], high->low res

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (B, C, *spatial_dims).

        Returns:
            out (torch.Tensor): Raw logits of shape (B, out_channels, *spatial_dims).
        """
        x = self._reshape_input(x)
        B, n_late_fusion, C, *spatial = x.shape

        if n_late_fusion != self.n_late_fusion:
            msg = (
                f"[{self.__class__.__name__}] Expected n_late_fusion={self.n_late_fusion}, "
                f"got {n_late_fusion}."
            )
            raise ValueError(msg)

        if C != self.in_channels:
            msg = (
                f"[{self.__class__.__name__}] in_channels={self.in_channels} but "
                f"input has {C} channels."
            )
            raise ValueError(msg)

        if len(spatial) != self.spatial_dims:
            msg = (
                f"[{self.__class__.__name__}] spatial_dims={self.spatial_dims} but "
                f"input has {len(spatial)} spatial dimensions."
            )
            raise ValueError(msg)

        # softmax weights across modalities per scale: [levels, n_late_fusion]
        alpha = torch.softmax(self.fusion_weights, dim=1)

        # running fused features per scale; fill lazily at first modality
        fused_in: torch.Tensor | None = None
        fused_feats: list[torch.Tensor | None] = [None] * (alpha.shape[0] - 1)

        # split input into chunks of size B and pass to encoder
        for m in range(n_late_fusion):
            in_m = x[:, m]  # (B, C, *spatial)
            # add zero scale (input img)
            w = alpha[0, m]
            fused_in = fused_in + w * in_m if fused_in is not None else w * in_m
            # encode and add rest
            feats_m = self._encode_one(in_m)
            for l in range(self.L):
                w = alpha[1 + l, m]
                prev = fused_feats[l]
                if prev is None:
                    fused_feats[l] = w * feats_m[l]
                else:
                    fused_feats[l] = prev + w * feats_m[l]
            del in_m, feats_m

        smoothed_feats = self.feats_decoder(fused_feats)
        out = self.head(smoothed_feats)

        return out
