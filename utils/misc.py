"""Miscellaneous utilities."""

from __future__ import annotations

from typing import Any, Sequence

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