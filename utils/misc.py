"""Miscellaneous utilities."""

from __future__ import annotations

__all__ = [
    "ensure_tuple_dim",
]

from typing import Any, Sequence, Literal, Optional
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