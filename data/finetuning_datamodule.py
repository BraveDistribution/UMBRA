from typing import Callable, Optional, Union, Literal, Sequence, Tuple, Set
from typing import cast
from pathlib import Path

import lightning.pytorch as pl
from sklearn.model_selection import train_test_split
from torch.utils.data   import DataLoader
from torch.utils.data.dataset import Dataset
from monai.data.utils import list_data_collate

from data.finetuning_dataset import FinetuningDataset
from utils.data import sample_subjects


class FinetuningDataModule(pl.LightningDataModule):  # type: ignore
    """
    General purpose data module for finetuning and testing.
    """

    def __init__(
        self,
        # Dataset
        data_dir: Union[str, Path],
        labels_dir: Optional[Union[str, Path]] = None,
        train_transforms: Optional[Callable] = None,
        val_transforms: Optional[Callable] = None,
        modalities: Optional[Sequence[str]] = None,
        scan_type: Literal["numpy", "nifti"] = "numpy",
        target: Literal["label", "mask", "combined"] = "mask",
        require_all_labels: bool = True,
        require_all_scans: bool = False,
        # Testing
        test_dir: Optional[Union[str, Path]] = None,
        test_labels_dir: Optional[Union[str, Path]] = None,
        train_val_split: float = 0.2,
        train_test_split: float = 0.2,
        # Data loading
        subset_train: Optional[float] = None,
        batch_size: int = 8,
        num_workers: int = 8,
        seed: int = 42,
    ):
        """
        Args:
            data_dir: Path to the data directory
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
            test_dir: Path to the test data directory; will override train/test split 
            test_labels_dir: Path to the test labels directory (mask/label files)
                All above settings (e.g., `scan_type`, `target`, `require_all_labels`, etc.)
                are applied to the test data as well
            train_test_split: Split ratio for train/test; ignored if `test_dir` is provided
            train_val_split: Split ratio for train/val
            subset_train: Subset ratio for train for train split; useful knob 
                for measuring few-shot performance
            batch_size: Batch size
            num_workers: Number of workers
            seed: Random seed
        """
        super().__init__()
        # Dataset settings
        self.data_dir = Path(data_dir)
        self.labels_dir = labels_dir
        self.modalities = modalities
        self.train_transforms = train_transforms
        self.val_transforms = val_transforms
        if scan_type not in ("numpy", "nifti"):
            raise ValueError(f"Invalid scan type: {scan_type}")
        self.scan_type = scan_type
        if target not in ("label", "mask", "combined"):
            raise ValueError(f"Invalid target: {target}")
        self.target = target
        self.require_all_labels = require_all_labels
        self.require_all_scans = require_all_scans
        self.test_dir = test_dir
        self.test_labels_dir = test_labels_dir
        self.train_val_split = train_val_split
        self.train_test_split = train_test_split
        self.subset_train = subset_train
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.seed = seed

        if not self.test_dir and self.train_test_split == 0:
            print("Warning: train_test_split is 0, so no test set will be created.")

        # Will get populated in `setup()`
        self.train_dataset: Optional[Dataset] = None
        self.val_dataset: Optional[Dataset] = None
        self.test_dataset: Optional[Dataset] = None
        self.predict_dataset: Optional[Dataset] = None

    def _get_train_val_test_ids(self) -> Tuple[Set[str], Set[str], Set[str]]:
        """
        Get train/val/test patient IDs.
        """
        print(f"Extracting patient IDs from {self.data_dir} to create train/val/test splits...")
        sub_ids = []
        for sub_dir in self.data_dir.iterdir():
            if sub_dir.is_dir() and sub_dir.name.startswith("sub_"):
                sub_id = sub_dir.name.replace("sub_", "")
                sub_ids.append(sub_id)

        # Sort subject IDs to ensure consistent ordering across all data modules
        # This is critical to prevent data leakage between train/val splits
        sub_ids = sorted(sub_ids)

        # Get test split
        if not self.test_dir:
            train_val, test = train_test_split(
                sub_ids, test_size=self.train_test_split, random_state=self.seed
            )
        else:
            train_val = sub_ids
            test = []

        # Extract subset of train/val if requested
        if self.subset_train:
            train_val, _ = sample_subjects(train_val, self.subset_train, self.seed)

        # Get train/val split
        train, val = train_test_split(
            train_val, test_size=self.train_val_split, random_state=self.seed
        )

        return set(train), set(val), set(test)

    def setup(self, stage: Optional[str]):
        """
        Get train/val/test/predict datasets.
        """
        train_ids, val_ids, test_ids = self._get_train_val_test_ids()

        if stage == "fit":
            self.train_dataset = FinetuningDataset(
                data_dir=self.data_dir,
                patients_included=train_ids,
                labels_dir=self.labels_dir,
                modalities=self.modalities,
                scan_type=cast(Literal["numpy", "nifti"], self.scan_type),
                target=cast(Literal["label", "mask", "combined"], self.target),
                transforms=self.train_transforms,
                require_all_labels=self.require_all_labels,
                require_all_scans=self.require_all_scans,
            )
        elif stage == "validate":
            self.val_dataset = FinetuningDataset(
            data_dir=self.data_dir,
            patients_included=val_ids,
            labels_dir=self.labels_dir,
            modalities=self.modalities,
            scan_type=cast(Literal["numpy", "nifti"], self.scan_type),
            target=cast(Literal["label", "mask", "combined"], self.target),
            transforms=self.val_transforms,
            require_all_labels=self.require_all_labels,
            require_all_scans=self.require_all_scans,
        )
        elif stage == "test":
            test_dir = self.data_dir if not self.test_dir else self.test_dir
            included_ids = test_ids if not self.test_dir else None
            self.test_dataset = FinetuningDataset(
                data_dir=test_dir,
                patients_included=included_ids,
                labels_dir=self.test_labels_dir,
                modalities=self.modalities,
                scan_type=cast(Literal["numpy", "nifti"], self.scan_type),
                target=cast(Literal["label", "mask", "combined"], self.target),
                transforms=self.val_transforms,
                require_all_labels=self.require_all_labels,
                require_all_scans=self.require_all_scans,
            )
        else: # stage == "predict"
            self.predict_dataset = FinetuningDataset(
            data_dir=self.data_dir,
            labels_dir=self.labels_dir,
            modalities=self.modalities,
            scan_type=cast(Literal["numpy", "nifti"], self.scan_type),
            target=cast(Literal["label", "mask", "combined"], self.target),
            transforms=self.val_transforms,
            require_all_labels=self.require_all_labels,
            require_all_scans=self.require_all_scans,
        )
    
    def train_dataloader(self):
        if self.train_dataset is None:
            raise ValueError(
                "Train dataset not found. Make sure setup('fit') is called first."
            )
        return DataLoader(
            cast(Dataset, self.train_dataset), 
            batch_size=self.batch_size, 
            shuffle=True, 
            num_workers=self.num_workers, 
            collate_fn=list_data_collate,
        )

    def val_dataloader(self):
        if self.val_dataset is None:
            raise ValueError(
                "Validation dataset not found. Make sure setup('validate') is called first."
            )
        return DataLoader(
            cast(Dataset, self.val_dataset), 
            batch_size=self.batch_size, 
            shuffle=False, 
            num_workers=self.num_workers, 
            collate_fn=list_data_collate,
        )
    
    def test_dataloader(self):
        if self.test_dataset is None:
            raise ValueError(
                "Test dataset not found. Make sure setup('test') is called first."
            )
        return DataLoader(
            cast(Dataset, self.test_dataset), 
            batch_size=self.batch_size, 
            shuffle=False, 
            num_workers=self.num_workers, 
            collate_fn=list_data_collate,
        )
    
    def predict_dataloader(self):
        if self.predict_dataset is None:
            raise ValueError(
                "Predict dataset not found. Make sure setup('predict') is called first."
            )
        return DataLoader(
            cast(Dataset, self.predict_dataset), 
            batch_size=self.batch_size, 
            shuffle=False, 
            num_workers=self.num_workers, 
            collate_fn=list_data_collate,
        )