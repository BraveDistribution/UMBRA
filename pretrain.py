from pathlib import Path
from typing import Optional, Sequence, Union

try:
    import pytorch_lightning as pl
    from pytorch_lightning.callbacks import ModelCheckpoint
    from pytorch_lightning.loggers import WandbLogger
except ImportError:
    pl = None  # type: ignore
    ModelCheckpoint = None  # type: ignore
    WandbLogger = None  # type: ignore

try:
    from fire import Fire
except ImportError:
    Fire = None  # type: ignore

from data.contrastive import ContrastiveDataModule
from models.foundation import ContrastiveTransformer


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
) -> None:
    save_dir: Union[str, Path] = Path(model_checkpoint_dir) / experiment_name

    if pl is None or ModelCheckpoint is None or WandbLogger is None:
        raise ImportError("pytorch_lightning package is required")

    import monai.transforms as T

    transforms = T.Compose(  # pyright: ignore[reportPrivateImportUsage]
        [
            # Ensure data is in the expected dictionary format
            T.AsDiscreted(  # pyright: ignore[reportPrivateImportUsage]
                keys=("vol1", "vol2"), argmax=False
            ),
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
    data_module = ContrastiveDataModule(
        data_dir=data_dir, transforms=transforms, batch_size=batch_size
    )
    if resume_from_checkpoint:
        model = ContrastiveTransformer.load_from_checkpoint(resume_from_checkpoint)
    else:
        model = ContrastiveTransformer(patch_size, learning_rate)
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
    if Fire is not None:
        Fire(train)
    else:
        print("Error: fire package not installed")
