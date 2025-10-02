from typing import Callable, Optional, Union

import os

from pathlib import Path

try:
    import pytorch_lightning as pl
    from pytorch_lightning.utilities.types import EVAL_DATALOADERS  # noqa: F401
except ImportError:
    # Fallback if pytorch_lightning is not installed
    pl = None  # type: ignore
    EVAL_DATALOADERS = None  # type: ignore
from sklearn.model_selection import train_test_split
from data.contrastive_dataset import ContrastivePatientDataset
from data.mae_dataset import MAEDataset
from torch.utils.data import DataLoader
import re


class ContrastiveDataModule(pl.LightningDataModule):  # type: ignore
    def __init__(
        self,
        data_dir: Union[str, Path],
        transforms: Optional[Callable] = None,
        batch_size: int = 10,
        include_mae: bool = False,
        mae_batch_size: Optional[int] = None,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.transforms = transforms
        self.batch_size = batch_size
        self.include_mae = include_mae
        self.mae_batch_size = (
            mae_batch_size if mae_batch_size is not None else batch_size
        )
        self.setup(None)

    def setup(self, stage):
        SESSION_RE = re.compile(
            r"sub_(?P<patient>\d+)_ses_(?P<session>\d+)_(?P<scan_type>.+)\.npy"
        )
        patient_ids = {
            int(match.group("patient"))
            for filename in os.listdir(self.data_dir)
            if (match := SESSION_RE.match(filename))
        }

        patient_ids = [str(i) for i in patient_ids]
        train_patients, val_patients = train_test_split(
            patient_ids, test_size=0.2, random_state=42
        )

        # Contrastive datasets (exclude scan_* files)
        self.train_dataset = ContrastivePatientDataset(
            data_dir=self.data_dir,
            patients_included=set(train_patients),
            transforms=self.transforms,
        )
        self.val_dataset = ContrastivePatientDataset(
            data_dir=self.data_dir,
            patients_included=set(val_patients),
            transforms=self.transforms,
        )

        # MAE datasets (include ALL scan types including scan_*)
        if self.include_mae:
            self.mae_train_dataset = MAEDataset(
                data_dir=self.data_dir,
                patients_included=set(train_patients),
                transforms=self.transforms,
            )
            self.mae_val_dataset = MAEDataset(
                data_dir=self.data_dir,
                patients_included=set(val_patients),
                transforms=self.transforms,
            )

    def train_dataloader(self):
        contrastive_loader = DataLoader(
            self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=32
        )

        if self.include_mae:
            mae_loader = DataLoader(
                self.mae_train_dataset,
                batch_size=self.mae_batch_size,
                shuffle=True,
                num_workers=32,
            )
            return [contrastive_loader, mae_loader]

        return contrastive_loader

    def val_dataloader(self):
        contrastive_loader = DataLoader(
            self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=32
        )

        if self.include_mae:
            mae_loader = DataLoader(
                self.mae_val_dataset,
                batch_size=self.mae_batch_size,
                shuffle=False,
                num_workers=32,
            )
            return [contrastive_loader, mae_loader]

        return contrastive_loader
