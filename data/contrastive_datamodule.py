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
from torch.utils.data import DataLoader


class ContrastiveDataModule(pl.LightningDataModule):  # type: ignore
    """Data module for Contrastive learning training only.

    This data module uses the ContrastivePatientDataset which excludes
    'scan_*' files and creates pairs of volumes from the same patient/session.
    """

    def __init__(
        self,
        data_dir: Union[str, Path],
        train_transforms: Optional[Callable] = None,
        val_transforms: Optional[Callable] = None,
        batch_size: int = 10,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.train_transforms = train_transforms
        self.val_transforms = val_transforms
        self.batch_size = batch_size
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
        self.train_dataset = ContrastivePatientDataset(
            data_dir=self.data_dir,
            patients_included=set(train_patients),
            transforms=self.train_transforms,
        )
        self.val_dataset = ContrastivePatientDataset(
            data_dir=self.data_dir,
            patients_included=set(val_patients),
            transforms=self.val_transforms,
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=32
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=32
        )
