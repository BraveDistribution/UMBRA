"""Miscellaneous utilities."""

from __future__ import annotations

__all__ = [
    "ensure_tuple_dim",
]

from typing import Any, Sequence, Literal, Iterable, Optional
import re
import math

def ensure_tuple_dim(
    value: Any | Sequence[Any],
    dim: int,
) -> tuple[Any, ...]:
    """Ensure that a value is a tuple of length `dim`."""
    if isinstance(value, (list, tuple)):
        if len(value) == dim:
            return tuple(value)
        else:
            raise ValueError(f"Expected tuple of length {dim}, got {len(value)}")
    elif isinstance(value, int | float):
        return (value,) * dim
    else:
        raise ValueError(f"Expected tuple or list, got {type(value)}")

def schedule_param(
    current: int,
    max_steps: int,
    start_frac: float,
    end_frac: float,
    start_val: float,
    end_val: float,
    mode: Literal["linear", "cosine", "constant"] = "linear",
) -> float:
    """
    Schedule a parameter value between `start_val` and `end_val` throughout training.
    Values are clamped to the `start_val` and `end_val` if outside the `start_frac` and 
    `end_frac` range.

    Args:
        current:     current step or epoch index
        max_steps:   total steps or epochs (defines the overall range)
        start_frac:  fraction (0–1) of `max_steps` where scheduling begins
        end_frac:    fraction (0–1) of `max_steps` where scheduling ends
        start_val:   value at start_frac
        end_val:     value at end_frac
        mode:        interpolation mode ("linear", "cosine", "constant")

    Returns:
        Interpolated parameter value (float) or None if outside range and clamp=False.
    """
    start = int(start_frac * max_steps)
    end = int(end_frac * max_steps)
    if end < start:
        raise ValueError("end_frac must be greater or equal to start_frac")

    # Clamp values
    if current <= start:
        return start_val
    if current >= end:
        return end_val

    # Normalized progress
    t = (current - start) / (end - start)

    if mode == "linear":
        return start_val + (end_val - start_val) * t
    elif mode == "cosine":
        c = 0.5 * (1 - math.cos(math.pi * t))
        return start_val + (end_val - start_val) * c
    elif mode == "constant":
        return start_val
    else:
        raise ValueError(f"Unknown mode '{mode}'")

def build_file_regex(
    stems: Optional[Iterable[str]] = None,
    exts: Iterable[str] = ("npy",),
    *,
    allow_index: bool = True,
    index_sep_class: str = r"[_-]",
    anchor: bool = True,
    flags: int = re.IGNORECASE,
) -> re.Pattern[str]:
    """
    Build a regex that matches files for the given stems and extensions.
    """
    # Handle "any modality" logic
    if not stems or any(m in ("*", "any") for m in stems):
        stem_pattern = r".+?"
    else:
        stem_list = [str(m).lower() for m in stems]
        if not stem_list:
            raise ValueError("stems must be a non-empty iterable")
        stem_pattern = "|".join(re.escape(m) for m in stem_list)

    # Handle extensions
    ext_list = sorted({str(e).lower().lstrip(".") for e in exts}, key=lambda s: -len(s))
    if not ext_list:
        raise ValueError("exts must be a non-empty iterable")

    ext_pattern = "|".join(re.escape(e) for e in ext_list)

    # Optional index (e.g., _1, -02)
    idx = rf"(?:{index_sep_class}?\d+)?" if allow_index else ""

    # Build full pattern
    pattern = rf"(?P<stem>{stem_pattern}){idx}\.(?P<ext>{ext_pattern})"
    if anchor:
        pattern = rf"^{pattern}$"

    return re.compile(pattern, flags)

def sync_dist_safe(obj) -> bool:
    """
    Return True if the attached Trainer is in DDP/FSDP mode (world_size > 1).

    Works for LightningModule, Callback, or anything else that has a .trainer.
    Falls back to False if we cannot decide.
    """
    try:
        trainer = getattr(obj, "trainer", None)
    except RuntimeError:
        return False
    
    if trainer is None:
        return False

    try:
        return getattr(trainer, "world_size", 1) > 1
    except AttributeError:
        return False