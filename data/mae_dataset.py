from typing import Any, Callable, Dict, List, Optional, Set, Union
import os
import numpy as np
from numpy.typing import NDArray
from torch.utils.data import Dataset
from pathlib import Path
import re

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


class MAEDataset(Dataset[Dict[str, NDArray[np.float32]]]):
    """Dataset for Masked Autoencoder (MAE) training.

    Returns individual volumes (no pairing) and includes ALL scan types,
    including 'scan_*' files that are excluded from contrastive learning.
    """

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
        self.volume_paths: List[str] = []
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

    def _load_volume_and_header(self, file: str) -> tuple[NDArray[np.float32], Any]:
        vol: NDArray[np.float32] = self._load_volume(file)
        header: Any = load_pickle(file[: -len(".npy")] + ".pkl")
        if np.isnan(vol).any() or np.isinf(vol).any():
            vol = np.nan_to_num(vol, nan=0.0, posinf=1.0, neginf=0.0, copy=True)
        return vol, header

    def populate_paths(self) -> None:
        """Populate list of all volume paths for included patients.

        Unlike ContrastivePatientDataset, this includes ALL scan types
        including 'scan_*' files.
        """
        self.volume_paths = []

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

            # Include ALL scan types (no exclusion of 'scan' files)
            full: str = os.path.join(self.data_dir, filename)
            self.volume_paths.append(full)

    def __len__(self) -> int:
        return len(self.volume_paths)

    def __getitem__(self, idx: int) -> Dict[str, NDArray[np.float32]]:
        vol_path: str = self.volume_paths[idx]
        vol: NDArray[np.float32]
        header: Any
        vol, header = self._load_volume_and_header(vol_path)

        data_dict: Dict[str, NDArray[np.float32]] = {"volume": vol}

        if self.transforms:
            data_dict = self.transforms(data_dict)

        return data_dict
