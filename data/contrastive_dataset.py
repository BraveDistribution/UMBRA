import os
import random
import re
from itertools import combinations
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union, Sequence

import numpy as np
from numpy.typing import NDArray
from torch.utils.data import Dataset

try:
    from batchgenerators.utilities.file_and_folder_operations import (  # type: ignore
        load_pickle,
    )
except ImportError:
    import pickle

    def load_pickle(file: str) -> Any:  # type: ignore
        with open(file, "rb") as f:
            return pickle.load(f)


from utils.misc import ensure_tuple_dim

# Pattern for hierarchical directory structure: sub_X/ses_Y/modality.npy
MODALITY_RE: re.Pattern[str] = re.compile(r"(?P<scan_type>.+?)(?:_\d+)?\.npy")


class ContrastivePatientDataset(Dataset[Dict[str, NDArray[np.float32]]]):
    def __init__(
        self,
        data_dir: Union[str, Path],
        patients_included: Set[str],
        transforms: Optional[
            Callable[[Dict[str, NDArray[np.float32]]], Dict[str, NDArray[np.float32]]]
        ] = None,
        patch_size: Union[int, Sequence[int]] = 96,
    ) -> None:
        """
        Args:
            data_dir: Path to the data directory
            patients_included: Set of patient IDs to include
            transforms: Transforms to apply to the data
            patch_size: Patch size to use for random crop
        """
        self.data_dir: Union[str, Path] = data_dir
        self.transforms: Optional[
            Callable[[Dict[str, NDArray[np.float32]]], Dict[str, NDArray[np.float32]]]
        ] = transforms
        self.patch_size = ensure_tuple_dim(patch_size, 3)
        self.patients_included: Set[str] = patients_included
        self.patients_sessions: Dict[str, Dict[str, List[str]]] = {}
        self.pairs: List[Dict[str, str]] = []
        self.populate_paths()

    def _load_volume(self, file: str) -> NDArray[np.float32]:
        path: str = (
            os.path.join(self.data_dir, file)
            if not file.startswith(str(self.data_dir))
            else file
        )
        vol: NDArray[Any]
        try:
            vol = np.load(path, "r")
        except ValueError:
            vol = np.load(path, allow_pickle=True)
        if len(vol.shape) == 3:
            vol = vol[np.newaxis, ...]
        return vol

    def _load_volume_and_header(self, file: str) -> Tuple[NDArray[np.float32], Any]:
        vol: NDArray[np.float32] = self._load_volume(file)
        yaml_path: str = file[: -len(".npy")] + ".yaml"
        pkl_path: str = file[: -len(".npy")] + ".pkl"

        header: Any = None
        if os.path.exists(yaml_path):
            import yaml
            with open(yaml_path, 'r') as f:
                header = yaml.safe_load(f)
        elif os.path.exists(pkl_path):
            header = load_pickle(pkl_path)

        if np.isnan(vol).any() or np.isinf(vol).any():
            vol = np.nan_to_num(vol, nan=0.0, posinf=1.0, neginf=0.0, copy=True)
        return vol, header

    def populate_paths(self) -> None:
        """Walk hierarchical directory structure: data_dir/sub_X/ses_Y/*.npy

        Structure:
        ├── sub_1
        │   └── ses_1
        │       ├── t1.npy
        │       ├── t1_2.npy
        │       └── flair.npy
        """
        self.patients_sessions = {}
        data_path: Path = Path(self.data_dir)

        # Walk through sub_X directories
        for patient_dir in data_path.iterdir():
            if not patient_dir.is_dir() or not patient_dir.name.startswith("sub_"):
                continue

            # Extract patient ID
            patient_id: str = patient_dir.name.replace("sub_", "")
            if patient_id not in self.patients_included:
                continue

            # Walk through ses_Y directories
            for session_dir in patient_dir.iterdir():
                if not session_dir.is_dir() or not session_dir.name.startswith("ses_"):
                    continue

                session_id: str = session_dir.name.replace("ses_", "")

                # Collect all .npy files in this session
                for npy_file in session_dir.glob("*.npy"):
                    filename: str = npy_file.name

                    # Parse modality name
                    m: Optional[re.Match[str]] = MODALITY_RE.match(filename)
                    if not m:
                        print(f"Warning: Unexpected filename: {filename}")
                        continue

                    scan_type: str = m.group("scan_type")

                    # Exclude scan files (scan, scan_pet, scan_ct, etc.)
                    if scan_type.startswith("scan"):
                        continue

                    full_path: str = str(npy_file)
                    self.patients_sessions.setdefault(patient_id, {}).setdefault(
                        session_id, []
                    ).append(full_path)

        self.pairs = []
        for patient, sessions in self.patients_sessions.items():
            for session, paths in sessions.items():
                if len(paths) >= 2:
                    for path1, path2 in combinations(paths, 2):
                        self.pairs.append(
                            {
                                "path1": path1,
                                "path2": path2,
                                "patient": patient,
                                "session": session,
                            }
                        )

    def __len__(self) -> int:
        return len(self.pairs)

    def _shared_random_crop(
        self,
        v1: NDArray[np.float32],
        v2: NDArray[np.float32],
        patch_size: Tuple[int, int, int],
    ) -> Tuple[NDArray[np.float32], NDArray[np.float32]]:
        """
        Apply the same random crop to both volumes so aligned.
        v1, v2: (C, D, H, W) arrays on the same grid.
        patch_size: tuple (pd, ph, pw) for depth, height, width.
        """
        pd: int
        ph: int
        pw: int
        pd, ph, pw = patch_size
        D: int
        H: int
        W: int
        D, H, W = v1.shape[-3:]

        # Random crop coordinates
        sd: int = 0 if D <= pd else random.randint(0, D - pd)
        sh: int = 0 if H <= ph else random.randint(0, H - ph)
        sw: int = 0 if W <= pw else random.randint(0, W - pw)

        # Apply crop to both
        v1c: NDArray[np.float32] = v1[..., sd : sd + pd, sh : sh + ph, sw : sw + pw]
        v2c: NDArray[np.float32] = v2[..., sd : sd + pd, sh : sh + ph, sw : sw + pw]

        # If any dim is smaller than patch, pad both identically
        def _pad_to_patch(x: NDArray[np.float32]) -> NDArray[np.float32]:
            Cd: int
            Ch: int
            Cw: int
            Cd, Ch, Cw = x.shape[-3:]
            pad_d: int = max(0, pd - Cd)
            pad_h: int = max(0, ph - Ch)
            pad_w: int = max(0, pw - Cw)
            if pad_d or pad_h or pad_w:
                x = np.pad(
                    x,
                    ((0, 0), (0, pad_d), (0, pad_h), (0, pad_w)),
                    mode="constant",
                    constant_values=0,
                )
            return x

        v1c = _pad_to_patch(v1c)
        v2c = _pad_to_patch(v2c)

        return v1c, v2c

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        pair_info: Dict[str, str] = self.pairs[idx]
        vol1: NDArray[np.float32]
        header1: Any
        vol1, header1 = self._load_volume_and_header(pair_info["path1"])
        vol2: NDArray[np.float32]
        header2: Any
        vol2, header2 = self._load_volume_and_header(pair_info["path2"])
        vol1, vol2 = self._shared_random_crop(vol1, vol2, patch_size=self.patch_size)

        data_dict: Dict[str, Any] = {
            "vol1": vol1,
            "vol2": vol2,
            "patient": pair_info["patient"],
            "session": pair_info["session"],
        }

        if self.transforms:
            # Transforms only apply to volumes, not metadata
            transformed = self.transforms({"vol1": vol1, "vol2": vol2})
            data_dict["vol1"] = transformed["vol1"]
            data_dict["vol2"] = transformed["vol2"]

        return data_dict
