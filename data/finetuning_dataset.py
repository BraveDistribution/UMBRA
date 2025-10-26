from typing import (
    Any, Callable, Dict, List, Optional, Set, Union, Tuple, Sequence, Literal,
)
from typing import cast
from pathlib import Path
import re

import numpy as np
from numpy.typing import NDArray
import torch
from torch.utils.data   import Dataset

from utils.data  import load_volume, load_label, get_file_ext
from utils.misc import build_file_regex
from transforms.composed import load_nifti


class FinetuningDataset(Dataset[Dict[str, Any]]):
    """
    General purpose dataset for finetuning.

    Supports both numpy and NIfTI files, as well as label types for all
    classification, regression and segmentation tasks.

    Numpy files are expected to be z-normalized, clipped to 0.5-99.5 percentile,
    and reoriented to RAS. NIfTI will be transformed in the same way on-the-fly.
    """
    def __init__(
        self,
        data_dir: Union[str, Path],
        patients_included: Optional[Set[str]] = None,
        *,
        labels_dir: Optional[Union[str, Path]] = None,
        transforms: Optional[
            Callable[
                [Union[Dict[str, NDArray], Dict[str, torch.Tensor]]], 
                Union[Dict[str, NDArray], Dict[str, torch.Tensor]]]
        ] = None,
        scan_type: Literal["numpy", "nifti"] = "numpy",
        modalities: Optional[Sequence[str]] = None,
        target: Literal["label", "mask", "combined"] = "mask",
        require_all_labels: bool = True,
        require_all_scans: bool = False,
        stage: Literal["train", "val", "test", "predict"] = "train",
    ) -> None:
        """
        Args:
            data_dir: Path to the data directory
            patients_included: Set of patient IDs to include
            labels_dir: Path to the labels directory (mask/label files)
            transforms: Transforms to apply to the data (other than IO)
            scan_type: Type of the scan. Possible values are:
                - "numpy": Numpy files.
                - "nifti": NIfTI files.
            modalities: List of modalities to restrict data collection
            target: Type of the target. Possible values are:
                - "label": Classification and regression tasks.
                     Gets a `label.txt` file as a label
                - "mask": Segmentation tasks. Gets a `mask.*` file as a label
                - "combined": Combined classification and segmentation tasks
                    Gets a `label.txt` file as a label and a `mask.*` file as a label
            require_all_labels: Skip session if a label is missing
            require_all_scans: Skip session if a scan is missing
            stage: Stage that dataset is being used for (train, val, test, predict);
                useful for logging.
        """
        self.data_dir: Path = Path(data_dir)
        self.labels_dir: Path= Path(labels_dir) if labels_dir else self.data_dir
        self.patients_included: Optional[Set[str]] = patients_included
        self.target = target
        self.scan_type = scan_type
        self.transforms: Optional[Callable[
            [Union[Dict[str, NDArray], Dict[str, torch.Tensor]]], 
            Union[Dict[str, NDArray], Dict[str, torch.Tensor]]]
        ] = transforms
        if stage not in ("train", "val", "test", "predict"):
            raise ValueError(f"Invalid stage: {stage}")
        self.stage = stage

        # Expected keys
        self.required_scan_keys = (
            set(modalities) if require_all_scans and modalities else set()
        )
        if require_all_labels and self.target == "label":
            self.required_label_keys = set(["label"])
        elif require_all_labels and self.target == "mask":
            self.required_label_keys = set(["mask"])
        elif require_all_labels and self.target == "combined":
            self.required_label_keys = set(["label", "mask"])
        else:
            self.required_label_keys = set()

        # Build regexes for collecting scans and labels
        exts=(".npy",) if scan_type == "numpy" else (".nii", ".nii.gz")
        self.mod_regex: re.Pattern[str] = build_file_regex(modalities, exts)
        self.mask_regex: re.Pattern[str] = build_file_regex(("mask", ), exts)
        self.label_regex: re.Pattern[str] = build_file_regex(("label", ), (".txt",))
        
        # Create list of data dictionaries
        self.data_entries: List[Dict[str, Any]] = []
        self.not_scan_keys: Set[str] = set(["patient_id", "session_id", "label"])
        self._populate_data()

    def _populate_data(self) -> None:
        """
        Populate the data entries by traversing through the data directory and creating a list of dictionaries, 
        unique for each subject-session combination. 

        Each entry has the following keys:
        - "patient_id": The ID of the subject.
        - "session_id": The ID of the session.
        - "label": The path to the label file (if applicable).
        - "mask": The path to the mask file (if applicable).
        - specific scan keys, unique per modality; e.g. "t1", "t2", "flair", etc.

        Excludes subjects that are not in the patients_included set (if provided).
        """
        print(f"Walking through {self.data_dir} to create dataset entries for {self.stage}...")
        for patient_dir in self.data_dir.iterdir():
            if not patient_dir.is_dir() or not patient_dir.name.startswith("sub_"):
                continue

            patient_id = patient_dir.name.replace("sub_", "")
            if self.patients_included and patient_id not in self.patients_included:
                continue

            for session_dir in patient_dir.iterdir():
                if not session_dir.is_dir() or not session_dir.name.startswith(
                    "ses_"
                ):
                    continue
                
                # Initialize session entry
                entry: Dict[str, Any] = {
                    "patient_id": patient_id,
                    "session_id": session_dir.name.replace("ses_", ""),
                }

                # Get labels
                labels_session_dir = self.labels_dir / session_dir.relative_to(self.data_dir)
                for file in labels_session_dir.iterdir():
                    if (
                        file.is_file() and 
                        self.target in ("label", "combined") and
                        self.label_regex.match(file.name)
                    ):
                        entry["label"] = str(file)
                    elif (
                        file.is_file() and 
                        self.target in ("mask", "combined") and
                        self.mask_regex.match(file.name)
                    ):
                        entry["mask"] = str(file)

                # Skip session if missing expected labels
                if self.required_label_keys:
                    missing = set()
                    for key in self.required_label_keys:
                        if key not in entry:
                            missing.add(key)
                    if missing:
                        print(f"Skipping session {session_dir.name} due to missing labels: "
                              f"{missing}")
                        continue
                    
                # Get all session scans (supports only .npy for now)
                for file in session_dir.iterdir():
                    if file.is_file() and self.mod_regex.match(file.name):
                        modality = file.name.replace(get_file_ext(file), "")
                        entry[modality] = str(file)

                # Skip session if missing expected scans
                if self.required_scan_keys:
                    missing = set()
                    for key in self.required_scan_keys:
                        if key not in entry:
                            missing.add(key)
                    if missing:
                        print(f"Skipping session {session_dir.name} due to missing scans: "
                              f"{missing}")
                        continue
                    
                self.data_entries.append(entry)
        
        print(f"Created {len(self.data_entries)} dataset entries for {self.stage}.")

    def __len__(self) -> int:
        return len(self.data_entries)

    def __getitem__(self, idx: int) -> List[Dict[str, Any]]:
        """
        Loads one sample:
        - all scan volumes (except 'mask') as float32
        - optional 'mask' as uint8
        - optional 'label' from label.txt (int/float)
        - passes through any metadata in self.not_scan_keys
        - applies self.transforms on the scans dict (if provided)
        - returns a list of dictionaries, each containing all scans as
          torch tensors (including masks), the label and any metadata.

        Use with MONAI's `list_data_collate` collate function for batching. 
        """
        entry = self.data_entries[idx]
        out: List[Dict[str, Any]] = []

        # scans-only dict for loading data and applying transforms
        scans: Union[List[Dict[str, Any]], Dict[str, Any]] = {
            k: v for k, v in entry.items() if k not in self.not_scan_keys
        }
        if self.scan_type == "nifti":
            mask_keys = ("mask",) if "mask" in scans else None
            scans = cast(
                Dict[str, Any], 
                load_nifti(keys=list(scans.keys()), mask_keys=mask_keys)(scans)
            )
        else:
            for key, value in scans.items():
                if key == "mask":
                    scans[key] = load_volume(value, dtype=np.uint8)
                else:
                    scans[key] = load_volume(value)
        if self.transforms is not None:
            scans = self.transforms(scans)

        # scans could be converted to list during transforms (e.g., multiple crops)
        # always convert to list for consistency, and let MONAI's `list_data_collate` handle collation. 
        if not isinstance(scans, list):
            scans = [scans]

        # Convert scans to torch tensors and add to output list
        for new_scans in cast(List[Dict[str, Any]], scans):
            new_entry: Dict[str, Any] = {}
            for key, arr in new_scans.items():
                if isinstance(arr, np.ndarray):
                    # Cast explicitly; avoids surprises and ensures contiguous tensors
                    if key == "mask":
                        new_entry[key] = torch.as_tensor(arr, dtype=torch.uint8)
                    else:
                        new_entry[key] = torch.as_tensor(arr, dtype=torch.float32)
                elif isinstance(arr, torch.Tensor):
                    # If it’s a MetaTensor, convert to plain tensor when available
                    new_entry[key] = arr.as_tensor() if hasattr(arr, "as_tensor") else arr # type: ignore[hasAttribute]
                # Skips other types (e.g., lists, strings, etc.) that could have been added during transforms;
                # unless in the original metadata.

            # Load label from label.txt (int/float)
            if "label" in entry:
                new_entry["label"] = load_label(entry["label"])

            # Pass any metadata that’s not part of scans/label
            for key, value in entry.items():
                if key not in new_scans.keys() and key != "label":
                    new_entry[key] = value

            out.append(new_entry)

        return out