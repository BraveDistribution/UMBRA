from typing import Any, Callable, Dict, List, Optional, Set, Union, Tuple, Sequence
import os
from pathlib import Path
import re

import numpy as np
from numpy.typing import NDArray
import torch
from torch.utils.data import Dataset

from utils import ensure_tuple_dim, load_volume


MODALITY_RE: re.Pattern[str] = re.compile(r"(?P<scan_type>.+?)(?:_\d+)?\.npy")


class MAEDataset(Dataset[Dict[str, Any]]):
    """Dataset for Masked Autoencoder (MAE) training.

    Returns individual volumes (no pairing) and includes ALL scan types,
    including 'scan_*' files that are excluded from contrastive learning.
    """

    def __init__(
        self,
        data_dir: Union[str, Path],
        patients_included: Set[str],
        transforms: Optional[
            Callable[[Dict[str, NDArray]], Union[Dict[str, NDArray], Dict[str, torch.Tensor]]]
        ] = None,
        exclude_contrastive_pairs: bool = False,
        input_size: Union[int, Sequence[int]] = 96,
    ) -> None:
        """
        Args:
            data_dir: Path to the data directory
            patients_included: Set of patient IDs to include
            transforms: Transforms to apply to the data
            exclude_contrastive_pairs: Whether to exclude contrastive pairs
            input_size: Input size to use for random crop
        """
        self.data_dir: Union[str, Path] = data_dir
        self.transforms: Optional[
            Callable[[Dict[str, NDArray]], Union[Dict[str, NDArray], Dict[str, torch.Tensor]]]
        ] = transforms
        self.patients_included: Set[str] = patients_included
        self.exclude_contrastive_pairs: bool = exclude_contrastive_pairs
        self.volume_paths: List[str] = []
        self.input_size: Tuple[int, int, int] = ensure_tuple_dim(input_size, 3)
        self.populate_paths()

    def populate_paths(self) -> None:
        """Walk hierarchical directory structure: data_dir/sub_X/ses_Y/*.npy

        Unlike ContrastivePatientDataset, this includes ALL scan types
        including 'scan_*' files.

        If exclude_contrastive_pairs=True, only includes:
        - Files with scan_type='scan' (excluded from contrastive)
        - Files that are alone (only 1 modality for that patient/session)
        """
        data_path: Path = Path(self.data_dir)
        patient_id: str
        session_id: str
        filename: str
        m: Optional[re.Match[str]]
        full_path: str

        print(f"Walking through {data_path} to create MAE dataset entries. This may take a while...")
        if self.exclude_contrastive_pairs:
            # Build structure to identify which files would be in contrastive pairs
            patients_sessions: Dict[str, Dict[str, List[Tuple[str, str]]]] = {}

            # Walk through sub_X/ses_Y directories
            for patient_dir in data_path.iterdir():
                if not patient_dir.is_dir() or not patient_dir.name.startswith("sub_"):
                    continue

                patient_id = patient_dir.name.replace("sub_", "")
                if patient_id not in self.patients_included:
                    continue

                for session_dir in patient_dir.iterdir():
                    if not session_dir.is_dir() or not session_dir.name.startswith(
                        "ses_"
                    ):
                        continue

                    session_id = session_dir.name.replace("ses_", "")

                    for npy_file in session_dir.glob("*.npy"):
                        filename = npy_file.name
                        m = MODALITY_RE.match(filename)
                        if not m:
                            continue

                        scan_type: str = m.group("scan_type")
                        full_path = str(npy_file)

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

                patient_id = patient_dir.name.replace("sub_", "")
                if patient_id not in self.patients_included:
                    continue

                for session_dir in patient_dir.iterdir():
                    if not session_dir.is_dir() or not session_dir.name.startswith(
                        "ses_"
                    ):
                        continue

                    for npy_file in session_dir.glob("*.npy"):
                        filename = npy_file.name
                        m = MODALITY_RE.match(filename)
                        if not m:
                            print(f"Warning: Unexpected filename: {filename}")
                            continue

                        # Include ALL scan types (no exclusion of 'scan' files)
                        full_path = str(npy_file)
                        self.volume_paths.append(full_path)

        print(f"Created {len(self.volume_paths)} MAE dataset entries.")

    def __len__(self) -> int:
        return len(self.volume_paths)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        vol_path: str = self.volume_paths[idx]

        # Load volume
        path: str = (
            os.path.join(self.data_dir, vol_path)
            if not vol_path.startswith(str(self.data_dir))
            else vol_path
        )
        vol = load_volume(path)
        data_dict: Dict[str, Any] = {"volume": vol}
        
        if self.transforms:
            data_dict = self.transforms(data_dict)
        else:
            # Default transforms
            from utils.spatial import random_crop
            data_dict["volume"] = random_crop(vol, self.input_size)

        # Convert to regular PyTorch tensor if not already
        # This handles both numpy arrays and MONAI MetaTensors
        for k in data_dict.keys():
            if not isinstance(data_dict[k], torch.Tensor):
                # Make a copy to ensure the array is writable
                data_dict[k] = torch.from_numpy(data_dict[k].copy()).float()
            elif hasattr(data_dict[k], "as_tensor"):
                # Convert MetaTensor to regular tensor
                data_dict[k] = data_dict[k].as_tensor() # type: ignore[hasAttribute]

        return data_dict