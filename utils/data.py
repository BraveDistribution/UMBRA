"""IO utilities."""

from __future__ import annotations

__all__ = [
    "load_volume", 
    "load_volume_and_header",
    "check_corrupted_files",
]

from typing import Any, Union, Optional, List, Tuple
from pathlib import Path
import os
import random

import numpy as np
from numpy.typing import NDArray, DTypeLike

try:
    from batchgenerators.utilities.file_and_folder_operations import (  # type: ignore
        load_pickle,
    )
except ImportError:
    import pickle

    def load_pickle(file: str) -> Any:  # type: ignore
        with open(file, "rb") as f:
            return pickle.load(f)

def load_volume(path: str, dtype: Optional[DTypeLike] = np.float32) -> NDArray[Any]:
    """Load a volume from a .npy file."""
    vol: NDArray[Any] = np.load(path, allow_pickle=True)
    if vol.ndim == 3:
        vol = vol[np.newaxis, ...] # (1,H,W,D)
    # ensure writable, correct dtype
    if not vol.flags.writeable:
        vol = np.array(vol, dtype=dtype, copy=True)
    else:
        vol = vol.astype(dtype, copy=False)
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

def load_label(path: Union[str, Path]) -> Union[int, float]:
    """Load a numeric label (int or float) from a .txt file."""
    text = Path(path).read_text().strip()
    try:
        # Try integer first for cleaner types
        return int(text)
    except ValueError:
        try:
            return float(text)
        except ValueError as e:
            raise ValueError(f"Invalid label in {path!s}: expected a numeric value, got {text!r}") from e

def check_corrupted_files(data_dir: Union[str, Path]) -> list[str]:
    data_dir = Path(data_dir)
    corrupted = []

    for npy_file in data_dir.rglob("*.npy"):
        try:
            vol = load_volume(str(npy_file))
            if vol.size == 0 or np.isnan(vol).all():
                corrupted.append(str(npy_file))
        except Exception as e:
            corrupted.append(str(npy_file))
            print(f"Error loading {npy_file}: {e}")

    print(f"Found {len(corrupted)} corrupted files")
    return corrupted

def get_file_ext(path: Union[Path, str]) -> str:
    """Get file extension from filepath with leading dot."""
    p = Path(path)
    suffixes = p.suffixes                    
    full_ext = ''.join(suffixes)              
    return full_ext

def sample_subjects(
    subjects: List[str],
    ratio: float,
    seed: int = 0
) -> Tuple[List[str], List[str]]:
    """
    Shuffle a list of subject IDs and split it by a given ratio.

    Args:
        subjects: List of subject IDs (e.g., ['sub001', 'sub002', ...]).
        ratio: Proportion to extract (0 < ratio < 1) — e.g., 0.2 means 20% held out.
        seed: Random seed for reproducibility.

    Returns:
        (selected, remaining): Tuple of two lists.
    """
    if not (0 < ratio < 1):
        raise ValueError(f"ratio must be between 0 and 1, got {ratio}")

    rng = random.Random(seed)
    shuffled = subjects[:]  # copy to avoid modifying original list
    rng.shuffle(shuffled)

    n_select = int(len(shuffled) * ratio)
    selected = shuffled[:n_select]
    remaining = shuffled[n_select:]

    return selected, remaining