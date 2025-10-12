from pathlib import Path
from typing import Optional, Sequence, Union

import pytorch_lightning as pl
from fire import Fire
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from data.combined_datamodule import CombinedDataModule
from data.contrastive_datamodule import ContrastiveDataModule
from data.mae_datamodule import MAEDataModule
from models.foundation import ContrastiveMAEPretrainer, MAEPretrainer


def _create_data_module(
    pretraining_mode: str,
    data_dir: Union[str, Path],
    mae_transforms,
    contrastive_transforms,
    batch_size: int,
    mae_batch_size: Optional[int],
) -> Union[MAEDataModule, ContrastiveDataModule, CombinedDataModule]:
    """Create appropriate data module based on pretraining mode.

    Args:
        pretraining_mode: One of 'mae_only', 'contrastive_only', or 'combined'
        data_dir: Directory containing training data
        mae_transforms: Transforms for MAE training
        contrastive_transforms: Transforms for contrastive learning
        batch_size: Batch size for training
        mae_batch_size: Optional separate batch size for MAE in combined mode

    Returns:
        Configured data module

    Raises:
        ValueError: If pretraining_mode is invalid
    """
    if pretraining_mode == "mae_only":
        return MAEDataModule(
            data_dir=data_dir, transforms=mae_transforms, batch_size=batch_size
        )
    elif pretraining_mode == "contrastive_only":
        return ContrastiveDataModule(
            data_dir=data_dir, transforms=contrastive_transforms, batch_size=batch_size
        )
    elif pretraining_mode == "combined":
        return CombinedDataModule(
            data_dir=data_dir,
            transforms=contrastive_transforms,  # For contrastive pairs (vol1, vol2)
            mae_transforms=mae_transforms,  # For MAE single volumes (volume key)
            batch_size=batch_size,
            mae_batch_size=mae_batch_size,
        )
    else:
        raise ValueError(
            f"Invalid pretraining_mode: {pretraining_mode}. "
            "Must be one of: 'mae_only', 'contrastive_only', 'combined'"
        )


def _create_or_load_model(
    pretraining_mode: str,
    resume_from_checkpoint: Optional[Union[str, Path]],
    patch_size: Sequence[int],
    learning_rate: float,
) -> Union[MAEPretrainer, ContrastiveMAEPretrainer]:
    """Create new model or load from checkpoint based on pretraining mode.

    Args:
        pretraining_mode: One of 'mae_only', 'contrastive_only', or 'combined'
        resume_from_checkpoint: Optional path to checkpoint to resume from
        patch_size: Size of patches for the model
        learning_rate: Learning rate for training

    Returns:
        Configured model instance
    """
    if resume_from_checkpoint:
        # Load from checkpoint (will auto-detect model class)
        if pretraining_mode == "mae_only":
            return MAEPretrainer.load_from_checkpoint(resume_from_checkpoint)
        else:
            return ContrastiveMAEPretrainer.load_from_checkpoint(resume_from_checkpoint)
    else:
        # Create new model based on pretraining mode
        if pretraining_mode == "mae_only":
            # Use MAEPretrainer for MAE-only training (cleaner, no contrastive components)
            return MAEPretrainer(
                patch_size=patch_size,
                learning_rate=learning_rate,
                pretraining_mode=pretraining_mode,
            )
        else:
            # Use ContrastiveMAEPretrainer for contrastive or combined mode
            return ContrastiveMAEPretrainer(
                patch_size=patch_size,
                learning_rate=learning_rate,
                pretraining_mode=pretraining_mode,
            )


def train(
    data_dir: Union[str, Path],
    model_checkpoint_dir: Union[str, Path] = "checkpoints",
    epochs: int = 100,
    patch_size: Sequence[int] = (96, 96, 96),
    batch_size: int = 10,
    learning_rate: float = 1e-4,
    accumulate_grad_batches: int = 3,
    experiment_name: str = "default_experiment",
    resume_from_checkpoint: Optional[Union[str, Path]] = None,
    pretraining_mode: str = "contrastive_only",
    mae_batch_size: Optional[int] = None,
) -> None:
    save_dir: Union[str, Path] = Path(model_checkpoint_dir) / experiment_name

    import monai.transforms as T

    # Define transforms based on pretraining mode
    # MAE-only transforms (single volume key)
    mae_transforms = T.Compose(  # pyright: ignore[reportPrivateImportUsage]
        [
            T.RandAdjustContrastd(  # pyright: ignore[reportPrivateImportUsage]
                keys="volume", prob=0.5, gamma=(0.8, 1.2)
            ),
            T.RandGibbsNoised(  # pyright: ignore[reportPrivateImportUsage]
                keys="volume", prob=0.3, alpha=(0.4, 0.8)
            ),
            T.RandGaussianNoised(  # pyright: ignore[reportPrivateImportUsage]
                keys="volume", prob=0.3, mean=0.0, std=0.1
            ),
            T.RandFlipd(  # pyright: ignore[reportPrivateImportUsage]
                keys="volume", prob=0.5, spatial_axis=0
            ),
            T.RandFlipd(  # pyright: ignore[reportPrivateImportUsage]
                keys="volume", prob=0.5, spatial_axis=1
            ),
            T.RandFlipd(  # pyright: ignore[reportPrivateImportUsage]
                keys="volume", prob=0.5, spatial_axis=2
            ),
            T.RandRotated(  # pyright: ignore[reportPrivateImportUsage]
                keys="volume",
                range_x=0.1,
                range_y=0.1,
                range_z=0.1,
                prob=0.8,
                mode="bilinear",
                padding_mode="zeros",
            ),
            T.RandZoomd(  # pyright: ignore[reportPrivateImportUsage]
                keys="volume",
                min_zoom=0.9,
                max_zoom=1.1,
                prob=0.8,
                mode="trilinear",
                align_corners=True,
            ),
            T.ToTensord(  # pyright: ignore[reportPrivateImportUsage]
                keys="volume"
            ),
        ]
    )

    # Contrastive transforms (vol1, vol2 keys) - used for contrastive_only and combined
    contrastive_transforms = T.Compose(  # pyright: ignore[reportPrivateImportUsage]
        [
            T.RandAdjustContrastd(  # pyright: ignore[reportPrivateImportUsage]
                keys="vol1", prob=0.5, gamma=(0.8, 1.2)
            ),
            T.RandAdjustContrastd(  # pyright: ignore[reportPrivateImportUsage]
                keys="vol2", prob=0.5, gamma=(0.8, 1.2)
            ),
            T.RandGibbsNoised(  # pyright: ignore[reportPrivateImportUsage]
                keys="vol1", prob=0.3, alpha=(0.4, 0.8)
            ),
            T.RandGibbsNoised(  # pyright: ignore[reportPrivateImportUsage]
                keys="vol2", prob=0.3, alpha=(0.4, 0.8)
            ),
            T.RandGaussianNoised(  # pyright: ignore[reportPrivateImportUsage]
                keys="vol1", prob=0.3, mean=0.0, std=0.1
            ),
            T.RandGaussianNoised(  # pyright: ignore[reportPrivateImportUsage]
                keys="vol2", prob=0.3, mean=0.0, std=0.1
            ),
            T.RandFlipd(  # pyright: ignore[reportPrivateImportUsage]
                keys=("vol1", "vol2"), prob=0.5, spatial_axis=0
            ),
            T.RandFlipd(  # pyright: ignore[reportPrivateImportUsage]
                keys=("vol1", "vol2"), prob=0.5, spatial_axis=1
            ),
            T.RandFlipd(  # pyright: ignore[reportPrivateImportUsage]
                keys=("vol1", "vol2"), prob=0.5, spatial_axis=2
            ),
            T.RandRotated(  # pyright: ignore[reportPrivateImportUsage]
                keys=("vol1", "vol2"),
                range_x=0.1,
                range_y=0.1,
                range_z=0.1,
                prob=0.8,
                mode=("bilinear", "bilinear"),
                padding_mode="zeros",
            ),
            T.RandZoomd(  # pyright: ignore[reportPrivateImportUsage]
                keys=("vol1", "vol2"),
                min_zoom=0.9,
                max_zoom=1.1,
                prob=0.8,
                mode=("trilinear", "trilinear"),
                align_corners=(True, True),
            ),
            T.ToTensord(  # pyright: ignore[reportPrivateImportUsage]
                keys=("vol1", "vol2")
            ),
        ]
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath=save_dir, filename="{epoch:02d}-{step}", every_n_train_steps=50
    )

    # Create data module based on pretraining mode
    data_module = _create_data_module(
        pretraining_mode=pretraining_mode,
        data_dir=data_dir,
        mae_transforms=mae_transforms,
        contrastive_transforms=contrastive_transforms,
        batch_size=batch_size,
        mae_batch_size=mae_batch_size,
    )

    # Create or load model based on pretraining mode
    model = _create_or_load_model(
        pretraining_mode=pretraining_mode,
        resume_from_checkpoint=resume_from_checkpoint,
        patch_size=patch_size,
        learning_rate=learning_rate,
    )

    wandb_logger = WandbLogger(
        project="PretrainingFOMO25New",
        name=experiment_name,
        entity="matejgazda-technical-university-of-kosice",
        log_model=True,
    )

    trainer = pl.Trainer(
        accelerator="gpu",
        logger=wandb_logger,
        callbacks=[checkpoint_callback],
        strategy="ddp",
        precision="bf16-mixed",
        accumulate_grad_batches=accumulate_grad_batches,
        max_epochs=epochs,
        log_every_n_steps=10,
        gradient_clip_val=1,
        gradient_clip_algorithm="norm",
    )

    if resume_from_checkpoint:
        trainer.fit(model, datamodule=data_module, ckpt_path=resume_from_checkpoint)
    else:
        trainer.fit(model, datamodule=data_module)


if __name__ == "__main__":
    Fire(train)
