import os
import random
import re
from itertools import combinations
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

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


SESSION_RE: re.Pattern[str] = re.compile(
    r"sub_(?P<patient>\d+)_ses_(?P<session>\d+)_(?P<scan_type>.+)\.npy"
)


class ContrastivePatientDataset(Dataset[Dict[str, NDArray[np.float32]]]):
    def __init__(
        self,
        data_dir: Union[str, Path],
        patients_included: Set[str],
        transforms: Optional[
            Callable[[Dict[str, NDArray[np.float32]]], Dict[str, NDArray[np.float32]]]
        ] = None,
    ) -> None:
        self.data_dir: Union[str, Path] = data_dir
        self.transforms: Optional[
            Callable[[Dict[str, NDArray[np.float32]]], Dict[str, NDArray[np.float32]]]
        ] = transforms
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
        header: Any = load_pickle(file[: -len(".npy")] + ".pkl")
        if np.isnan(vol).any() or np.isinf(vol).any():
            vol = np.nan_to_num(vol, nan=0.0, posinf=1.0, neginf=0.0, copy=True)
        return vol, header

    def populate_paths(self) -> None:
        """patients_sessions: {patient: {session: [full_paths...]}}"""
        self.patients_sessions = {}

        for filename in os.listdir(self.data_dir):
            if not filename.endswith(".npy"):
                continue

            # Extract patient ID for filtering
            split_parts: List[str] = filename.split("_")
            if len(split_parts) <= 1:
                continue

            patient_id: str = split_parts[1]
            if patient_id not in self.patients_included:
                continue

            m: Optional[re.Match[str]] = SESSION_RE.match(filename)
            if not m:
                print(f"Warning: Unexpected filename: {filename}")
                continue

            patient: str = m.group("patient")
            session: str = m.group("session")
            scan_type: str = m.group("scan_type")

            # Exclude files with scan_type "scan"
            if scan_type == "scan":
                continue

            full: str = os.path.join(self.data_dir, filename)
            self.patients_sessions.setdefault(patient, {}).setdefault(
                session, []
            ).append(full)

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
        patch_size: Tuple[int, int, int] = (96, 96, 96),
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

    def __getitem__(self, idx: int) -> Dict[str, NDArray[np.float32]]:
        pair_info: Dict[str, str] = self.pairs[idx]
        vol1: NDArray[np.float32]
        header1: Any
        vol1, header1 = self._load_volume_and_header(pair_info["path1"])
        vol2: NDArray[np.float32]
        header2: Any
        vol2, header2 = self._load_volume_and_header(pair_info["path2"])
        vol1, vol2 = self._shared_random_crop(vol1, vol2)

        data_dict: Dict[str, NDArray[np.float32]] = {"vol1": vol1, "vol2": vol2}

        if self.transforms:
            data_dict = self.transforms(data_dict)

        return data_dict
