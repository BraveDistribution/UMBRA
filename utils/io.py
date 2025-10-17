"""IO utilities."""

from __future__ import annotations

__all__ = [
    "load_volume", 
    "load_volume_and_header",
]

import os
import numpy as np
from numpy.typing import NDArray
from typing import Any

try:
    from batchgenerators.utilities.file_and_folder_operations import (  # type: ignore
        load_pickle,
    )
except ImportError:
    import pickle

    def load_pickle(file: str) -> Any:  # type: ignore
        with open(file, "rb") as f:
            return pickle.load(f)

def load_volume(path: str) -> NDArray[np.float32]:
    """Load a volume from a .npy file."""
    vol: NDArray[Any] = np.load(path, allow_pickle=True)
    if vol.ndim == 3:
        vol = vol[np.newaxis, ...] # (1,H,W,D)
    # ensure writable, correct dtype
    if not vol.flags.writeable:
        vol = np.array(vol, dtype=np.float32, copy=True)
    else:
        vol = vol.astype(np.float32, copy=False)
    return vol

def load_volume_and_header(file: str) -> tuple[NDArray[np.float32], Any]:
    vol: NDArray[np.float32] = load_volume(file)
    yaml_path: str = file[: -len(".npy")] + ".yaml"
    pkl_path: str = file[: -len(".npy")] + ".pkl"

    header: Any = None
    if os.path.exists(yaml_path):
        import yaml

        with open(yaml_path, "r") as f:
            header = yaml.safe_load(f)
    elif os.path.exists(pkl_path):
        header = load_pickle(pkl_path)

    if np.isnan(vol).any() or np.isinf(vol).any():
        vol = np.nan_to_num(vol, nan=0.0, posinf=1.0, neginf=0.0, copy=True)
    return vol, header