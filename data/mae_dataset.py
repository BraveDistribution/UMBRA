from typing import Any, Callable, Dict, List, Optional, Set, Union, Tuple
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


MODALITY_RE: re.Pattern[str] = re.compile(r"(?P<scan_type>.+?)(?:_\d+)?\.npy")


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
        exclude_contrastive_pairs: bool = False,
    ) -> None:
        self.data_dir: Union[str, Path] = data_dir
        self.transforms: Optional[
            Callable[[Dict[str, NDArray[np.float32]]], Dict[str, NDArray[np.float32]]]
        ] = transforms
        self.patients_included: Set[str] = patients_included
        self.exclude_contrastive_pairs: bool = exclude_contrastive_pairs
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

    def populate_paths(self) -> None:
        """Walk hierarchical directory structure: data_dir/sub_X/ses_Y/*.npy

        Unlike ContrastivePatientDataset, this includes ALL scan types
        including 'scan_*' files.

        If exclude_contrastive_pairs=True, only includes:
        - Files with scan_type='scan' (excluded from contrastive)
        - Files that are alone (only 1 modality for that patient/session)
        """
        self.volume_paths = []
        data_path: Path = Path(self.data_dir)

        if self.exclude_contrastive_pairs:
            # Build structure to identify which files would be in contrastive pairs
            patients_sessions: Dict[str, Dict[str, List[Tuple[str, str]]]] = {}

            # Walk through sub_X/ses_Y directories
            for patient_dir in data_path.iterdir():
                if not patient_dir.is_dir() or not patient_dir.name.startswith("sub_"):
                    continue

                patient_id: str = patient_dir.name.replace("sub_", "")
                if patient_id not in self.patients_included:
                    continue

                for session_dir in patient_dir.iterdir():
                    if not session_dir.is_dir() or not session_dir.name.startswith(
                        "ses_"
                    ):
                        continue

                    session_id: str = session_dir.name.replace("ses_", "")

                    for npy_file in session_dir.glob("*.npy"):
                        filename: str = npy_file.name
                        m: Optional[re.Match[str]] = MODALITY_RE.match(filename)
                        if not m:
                            continue

                        scan_type: str = m.group("scan_type")
                        full_path: str = str(npy_file)

                        # Group files by patient/session, tracking scan_type
                        patients_sessions.setdefault(patient_id, {}).setdefault(
                            session_id, []
                        ).append((full_path, scan_type))

            # Now filter: only include 'scan' files OR files that are alone
            for patient, sessions in patients_sessions.items():
                for session, files_info in sessions.items():
                    # Separate scan files from non-scan files
                    # scan_type starts with "scan" for files like scan_pet, scan_ct, etc.
                    non_scan_files: List[Tuple[str, str]] = [
                        (path, st) for path, st in files_info if not st.startswith("scan")
                    ]
                    scan_files: List[Tuple[str, str]] = [
                        (path, st) for path, st in files_info if st.startswith("scan")
                    ]

                    # Add all 'scan' files (always excluded from contrastive)
                    for path, _ in scan_files:
                        self.volume_paths.append(path)

                    # Add non-scan files ONLY if there's just 1 (can't form pairs)
                    if len(non_scan_files) == 1:
                        self.volume_paths.append(non_scan_files[0][0])
                    # If len >= 2, these would be in contrastive pairs, so exclude

        else:
            # Original behavior: include ALL files
            for patient_dir in data_path.iterdir():
                if not patient_dir.is_dir() or not patient_dir.name.startswith("sub_"):
                    continue

                patient_id: str = patient_dir.name.replace("sub_", "")
                if patient_id not in self.patients_included:
                    continue

                for session_dir in patient_dir.iterdir():
                    if not session_dir.is_dir() or not session_dir.name.startswith(
                        "ses_"
                    ):
                        continue

                    for npy_file in session_dir.glob("*.npy"):
                        filename: str = npy_file.name
                        m: Optional[re.Match[str]] = MODALITY_RE.match(filename)
                        if not m:
                            print(f"Warning: Unexpected filename: {filename}")
                            continue

                        # Include ALL scan types (no exclusion of 'scan' files)
                        full_path: str = str(npy_file)
                        self.volume_paths.append(full_path)

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
