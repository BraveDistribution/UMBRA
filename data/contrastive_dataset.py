from itertools import combinations
from pathlib import Path
import re
from typing import Any, Callable, Dict, List, Optional, Set, Literal, Union, Sequence

import numpy as np
from numpy.typing import NDArray
import torch
from torch.utils.data   import Dataset

from utils import (
    ensure_tuple_dim,
    load_volume,
)

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
        input_size: Union[int, Sequence[int]] = 96,
        contrastive_mode: Literal["regular", "modality_pairs"] = "modality_pairs",
    ) -> None:
        """
        Args:
            data_dir: Path to the data directory
            patients_included: Set of patient IDs to include
            transforms: Transforms to apply to the data
            input_size: Input size to use for random crop
            contrastive_mode: Mode to use for contrastive learning:
             - "regular": Augmented views for positive pairs.
             - "modality_pairs": Use different modalities as positive pairs.
        """
        self.data_dir: Union[str, Path] = data_dir
        self.transforms: Optional[
            Callable[[Dict[str, NDArray[np.float32]]], Dict[str, NDArray[np.float32]]]
        ] = transforms
        self.input_size = ensure_tuple_dim(input_size, 3)
        self.patients_included: Set[str] = patients_included
        self.contrastive_mode: Literal["regular", "modality_pairs"] = contrastive_mode
        self.patients_sessions: Dict[str, Dict[str, List[str]]] = {}
        self.pairs: List[Dict[str, str]] = []
        self._populate_paths()

    def _populate_paths(self) -> None:
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

        print(f"Walking through {data_path} to create contrastive pairs. This may take a while...")
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
                # Same inclusion criteria to ensure fair comparison between methods
                if len(paths) >= 2:
                    if self.contrastive_mode == "modality_pairs":
                        for path1, path2 in combinations(paths, 2):
                            self.pairs.append(
                                {
                                    "path1": path1,
                                    "path2": path2,
                                    "patient": patient,
                                    "session": session,
                                }
                            )
                    else:
                        for path in paths:
                            self.pairs.append(
                                {
                                    "path1": path,
                                    "path2": path,
                                    "patient": patient,
                                    "session": session,
                                }
                            )

        print(f"Created {len(self.pairs)} contrastive pairs.")

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        pair_info: Dict[str, str] = self.pairs[idx]
        vol1: NDArray[np.float32]
        vol2: NDArray[np.float32]
        vol1 = load_volume(pair_info["path1"])
        vol2 = load_volume(pair_info["path2"])

        data_dict: Dict[str, Any] = {
            "vol1": vol1,
            "vol2": vol2,
            "patient": pair_info["patient"],
            "session": pair_info["session"],
        }

        if self.transforms:
            # Transforms only applied to volumes for safety
            transformed = self.transforms({"vol1": vol1, "vol2": vol2})
            # Update only the volume keys (and any recon keys)
            data_dict["vol1"] = transformed["vol1"]
            data_dict["vol2"] = transformed["vol2"]
            # Add reconstruction targets if they exist
            if "vol1_recon" in transformed:
                data_dict["vol1_recon"] = transformed["vol1_recon"]
            if "vol2_recon" in transformed:
                data_dict["vol2_recon"] = transformed["vol2_recon"]
        else:
            # Default transforms
            from utils.spatial import shared_random_crop
            data_dict["vol1"], data_dict["vol2"] = (
                shared_random_crop(vol1, vol2, self.input_size)
            )

        # Convert to regular PyTorch tensor if not already
        # This handles both numpy arrays and MONAI MetaTensors
        for k in data_dict.keys():
            # Skip metadata keys
            if isinstance(data_dict[k], torch.Tensor):
                if hasattr(data_dict[k], "as_tensor"):
                    # Convert MetaTensor to regular tensor
                    data_dict[k] = data_dict[k].as_tensor() # type: ignore[hasAttribute]
            elif isinstance(data_dict[k], np.ndarray):
                # Make a copy to ensure the array is writable
                data_dict[k] = torch.from_numpy(data_dict[k].copy()).float()
            # Skip other types (e.g., lists, strings, etc.) - these are metadata

        return data_dict
