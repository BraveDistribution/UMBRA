from typing import Any, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from monai.networks.nets.swin_unetr import SwinTransformer as SwinViT


def ensure_tuple_dim(
    value: Any | Sequence[Any],
    dim: int,
) -> tuple[Any, ...]:
    """Ensure that a value is a tuple of length `dim`."""
    if isinstance(value, (list, tuple)):
        if len(value) == dim:
            return tuple(value)
        else:
            msg = f"Expected tuple of length {dim}, got {len(value)}"
            raise ValueError(msg)
    elif isinstance(value, int | float):
        return (value,) * dim
    else:
        msg = f"Expected tuple or list, got {type(value)}"
        raise ValueError(msg)


class ConvBNAct3d(nn.Module):
    """
    Convolutional block with normalization and activation.
    """

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        k: int = 3,
        s: int = 1,
        p: int = 1,
        norm: str = "instance",
        *,
        act_kwargs: dict[str, Any] | None = None,
    ):
        if act_kwargs is None:
            act_kwargs = {}

        super().__init__()
        if norm == "instance":
            Norm = nn.InstanceNorm3d
        elif norm == "group":
            Norm = lambda c: nn.GroupNorm(num_groups=min(8, c), num_channels=c)
        else:
            Norm = nn.BatchNorm3d

        self.block = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=k, stride=s, padding=p, bias=False),
            Norm(out_ch),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.block(x)


class FPNLightDecoder3D(nn.Module):
    """
    Top-down FPN with narrow width and one 3x3 smooth per level.
    in_chans: channels of encoder features [c1,c2,c3,c4,c5] (f1 highest res)
    width: lateral width (e.g., 32 or 64)
    """

    def __init__(
        self,
        in_feats: Sequence[int],
        in_chans: int,
        out_channels: int,
        width: int = 32,
        norm: str = "instance",
    ):
        super().__init__()
        assert len(in_feats) == 5, "Need 5 scales [f1..f5]"
        f1, f2, f3, f4, f5 = in_feats

        self.in_stem = ConvBNAct3d(in_chans, width, k=3, p=1, norm=norm)

        self.lat1 = nn.Conv3d(f1, width, kernel_size=1, bias=False)
        self.lat2 = nn.Conv3d(f2, width, kernel_size=1, bias=False)
        self.lat3 = nn.Conv3d(f3, width, kernel_size=1, bias=False)
        self.lat4 = nn.Conv3d(f4, width, kernel_size=1, bias=False)
        self.lat5 = nn.Conv3d(f5, width, kernel_size=1, bias=False)

        # one light smooth per pyramid level after lateral+topdown add
        self.smooth4 = ConvBNAct3d(width, width, k=3, p=1, norm=norm)
        self.smooth3 = ConvBNAct3d(width, width, k=3, p=1, norm=norm)
        self.smooth2 = ConvBNAct3d(width, width, k=3, p=1, norm=norm)
        self.smooth1 = ConvBNAct3d(width, width, k=3, p=1, norm=norm)
        self.smooth0 = ConvBNAct3d(width, width, k=3, p=1, norm=norm)

        # head on the finest map (keeps params tiny)
        self.head = nn.Conv3d(width, out_channels, kernel_size=1)

    def _upsample_to(self, x: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        return F.interpolate(
            x, size=ref.shape[-3:], mode="trilinear", align_corners=False
        )

    def forward(self, x: torch.Tensor, feats: list[torch.Tensor]) -> torch.Tensor:
        # feats: [f1,f2,f3,f4,f5] high->low res
        if len(feats) != 5:
            msg = f"[{self.__class__.__name__}] Expected 5 features, got {len(feats)}."
            raise ValueError(msg)

        f1, f2, f3, f4, f5 = feats
        p5 = self.lat5(f5)  # bottleneck
        p4 = self.smooth4(self.lat4(f4) + self._upsample_to(p5, f4))
        p3 = self.smooth3(self.lat3(f3) + self._upsample_to(p4, f3))
        p2 = self.smooth2(self.lat2(f2) + self._upsample_to(p3, f2))
        p1 = self.smooth1(self.lat1(f1) + self._upsample_to(p2, f1))
        p0 = self.smooth0(self.in_stem(x) + self._upsample_to(p1, x))
        return self.head(p0)  # logits at full res


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
        norm: str = "instance",
    ):
        super().__init__()

        extra_swin_kwargs = extra_swin_kwargs or {}

        self.encoder = SwinViT(
            in_chans=in_channels,
            embed_dim=feature_size,
            depths=depths,
            num_heads=num_heads,
            window_size=ensure_tuple_dim(window_size, spatial_dims),
            patch_size=ensure_tuple_dim(patch_size, spatial_dims),
            use_v2=use_v2,
            **extra_swin_kwargs,
        )

        # MONAI's `SwinTransformer` returns input after patch_embed and 4-level feature maps
        in_feats = [feature_size * 2**i for i in range(len(depths) + 1)]
        self.L = len(in_feats)

        self.decoder = FPNLightDecoder3D(
            in_feats=in_feats,
            in_chans=in_channels,
            out_channels=out_channels,
            width=width,
            norm=norm,
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
            torch.Tensor: Raw logits of shape (B, num_classes).
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
                fused_feats[l] = (
                    fused_feats[l] + w * feats_m[l]
                    if fused_feats[l]  # type: ignore
                    is not None
                    else w * feats_m[l]
                )
            del in_m, feats_m

        return self.decoder(fused_in, fused_feats)
