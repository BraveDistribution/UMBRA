from typing import Callable, Optional, Union

from pathlib import Path

try:
    import lightning.pytorch as pl
    from lightning.pytorch.utilities.types import EVAL_DATALOADERS  # noqa: F401
except ImportError:
    # Fallback if lightning.pytorch is not installed
    pl = None  # type: ignore
    EVAL_DATALOADERS = None  # type: ignore
from sklearn.model_selection import train_test_split
from data.contrastive_dataset import ContrastivePatientDataset
from data.mae_dataset import MAEDataset
from torch.utils.data import DataLoader


class CombinedDataModule(pl.LightningDataModule):  # type: ignore
    """Data module for combined MAE + Contrastive learning training.

    This data module provides both datasets WITHOUT overlap:
    - ContrastivePatientDataset: pairs of volumes from same patient/session (excludes 'scan_*')
    - MAEDataset: individual volumes that are NOT in contrastive pairs
      (includes 'scan_*' files + single modality files only)

    This ensures no image is used twice across both datasets.

    Returns two dataloaders as a list for training and validation.
    """

    def __init__(
        self,
        data_dir: Union[str, Path],
        batch_size: int = 10,
        mae_batch_size: Optional[int] = None,
        mae_train_transforms: Optional[Callable] = None,
        mae_val_transforms: Optional[Callable] = None,
        contrastive_train_transforms: Optional[Callable] = None,
        contrastive_val_transforms: Optional[Callable] = None,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.contrastive_train_transforms = contrastive_train_transforms  # Contrastive transforms (vol1, vol2 keys)
        self.contrastive_val_transforms = contrastive_val_transforms  # Contrastive transforms (vol1, vol2 keys)
        self.mae_train_transforms = mae_train_transforms  # MAE transforms (volume key)
        self.mae_val_transforms = mae_val_transforms  # MAE transforms (volume key)
        self.batch_size = batch_size
        self.mae_batch_size = (
            mae_batch_size if mae_batch_size is not None else batch_size
        )
        self.setup(None)

    def setup(self, stage: Optional[str]):
        # Extract patient IDs from hierarchical directory structure
        data_path = Path(self.data_dir)
        patient_ids = []

        for patient_dir in data_path.iterdir():
            if patient_dir.is_dir() and patient_dir.name.startswith("sub_"):
                patient_id = patient_dir.name.replace("sub_", "")
                patient_ids.append(patient_id)

        # Sort patient IDs to ensure consistent ordering across all data modules
        # This is critical to prevent data leakage between train/val splits
        patient_ids = sorted(patient_ids)

        train_patients, val_patients = train_test_split(
            patient_ids, test_size=0.2, random_state=42
        )

        # Contrastive datasets (exclude scan_* files)
        self.contrastive_train_dataset = ContrastivePatientDataset(
            data_dir=self.data_dir,
            patients_included=set(train_patients),
            transforms=self.contrastive_train_transforms,
        )
        self.contrastive_val_dataset = ContrastivePatientDataset(
            data_dir=self.data_dir,
            patients_included=set(val_patients),
            transforms=self.contrastive_val_transforms,
        )

        # MAE datasets (exclude files that are in contrastive pairs)
        # Only includes: 'scan' files + single modality files
        self.mae_train_dataset = MAEDataset(
            data_dir=self.data_dir,
            patients_included=set(train_patients),
            transforms=self.mae_train_transforms,  # Use MAE-specific transforms
            exclude_contrastive_pairs=True,
        )
        self.mae_val_dataset = MAEDataset(
            data_dir=self.data_dir,
            patients_included=set(val_patients),
            transforms=self.mae_val_transforms,  # Use MAE-specific transforms
            exclude_contrastive_pairs=True,
        )

    def train_dataloader(self):
        contrastive_loader = DataLoader(
            self.contrastive_train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=32,
        )

        mae_loader = DataLoader(
            self.mae_train_dataset,
            batch_size=self.mae_batch_size,
            shuffle=True,
            num_workers=32,
        )

        return [contrastive_loader, mae_loader]

    def val_dataloader(self):
        contrastive_loader = DataLoader(
            self.contrastive_val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=32,
        )

        mae_loader = DataLoader(
            self.mae_val_dataset,
            batch_size=self.mae_batch_size,
            shuffle=False,
            num_workers=32,
        )

        return [contrastive_loader, mae_loader]
