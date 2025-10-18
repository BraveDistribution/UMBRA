from pathlib import Path
from typing import Optional, Sequence, Union, Callable, Optional, Literal
from typing import cast

import lightning.pytorch as pl
from fire import Fire
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.strategies import DDPStrategy

from data.combined_datamodule import CombinedDataModule
from data.contrastive_datamodule import ContrastiveDataModule
from data.mae_datamodule import MAEDataModule
from models.foundation import ContrastiveMAEPretrainer, MAEPretrainer
from transforms.composed import get_mae_transforms, get_contrastive_transforms
from callbacks.monitor import LogLR, LogGradNorm


def _create_data_module(
    pretraining_mode: Literal["mae_only", "contrastive_only", "combined"],
    data_dir: Union[str, Path],
    mae_train_transforms: Optional[Callable],
    mae_val_transforms: Optional[Callable],
    contrastive_train_transforms: Optional[Callable],
    contrastive_val_transforms: Optional[Callable],
    contrastive_mode: Literal["regular", "modality_pairs"],
    batch_size: int,
    num_workers: int,
    input_size: Union[int, Sequence[int]],
    mae_batch_size: Optional[int],
    seed: int,
) -> Union[MAEDataModule, ContrastiveDataModule, CombinedDataModule]:
    """Create appropriate data module based on pretraining mode.

    Args:
        pretraining_mode: One of 'mae_only', 'contrastive_only', or 'combined'
        data_dir: Directory containing training data
        mae_train_transforms: Transforms for MAE training
        mae_val_transforms: Transforms for MAE validation
        contrastive_train_transforms: Transforms for contrastive learning training
        contrastive_val_transforms: Transforms for contrastive learning validation
        contrastive_mode: Mode to use for contrastive learning
        batch_size: Batch size for training
        num_workers: Number of workers for data loading
        mae_batch_size: Optional separate batch size for MAE in combined mode
        input_size: Input image dimensions
        seed: Random seed for reproducibility

    Returns:
        Configured data module

    Raises:
        ValueError: If pretraining_mode is invalid
    """
    if pretraining_mode == "mae_only":
        return MAEDataModule(
            data_dir=data_dir, 
            train_transforms=mae_train_transforms, 
            val_transforms=mae_val_transforms, 
            batch_size=batch_size,
            num_workers=num_workers,
            input_size=input_size,
            seed=seed,
        )
    elif pretraining_mode == "contrastive_only":
        return ContrastiveDataModule(
            data_dir=data_dir, 
            train_transforms=contrastive_train_transforms, 
            val_transforms=contrastive_val_transforms, 
            batch_size=batch_size,
            num_workers=num_workers,
            contrastive_mode=contrastive_mode,
            input_size=input_size,
            seed=seed,
        )
    elif pretraining_mode == "combined":
        return CombinedDataModule(
            data_dir=data_dir,
            contrastive_train_transforms=contrastive_train_transforms,  # For contrastive pairs (vol1, vol2)
            contrastive_val_transforms=contrastive_val_transforms,  # For contrastive pairs (vol1, vol2)
            contrastive_mode=contrastive_mode,
            mae_train_transforms=mae_train_transforms,  # For MAE single volumes (volume key)
            mae_val_transforms=mae_val_transforms,  # For MAE single volumes (volume key)
            batch_size=batch_size,
            mae_batch_size=mae_batch_size,
            num_workers=num_workers,
            input_size=input_size,
            seed=seed,
        )
    else:
        raise ValueError(
            f"Invalid pretraining_mode: {pretraining_mode}. "
            "Must be one of: 'mae_only', 'contrastive_only', 'combined'"
        )


def _create_or_load_model(
    pretraining_mode: Literal["mae_only", "contrastive_only", "combined"],
    resume_from_checkpoint: Optional[Union[str, Path]],
    input_size: Union[int, Sequence[int]],
    learning_rate: float,
) -> Union[MAEPretrainer, ContrastiveMAEPretrainer]:
    """Create new model or load from checkpoint based on pretraining mode.

    Args:
        pretraining_mode: One of 'mae_only', 'contrastive_only', or 'combined'
        resume_from_checkpoint: Optional path to checkpoint to resume from
        input_size: Input image dimensions
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
                input_size=input_size,
                learning_rate=learning_rate,
            )
        else:
            # Use ContrastiveMAEPretrainer for contrastive or combined mode
            return ContrastiveMAEPretrainer(
                input_size=input_size,
                learning_rate=learning_rate,
                pretraining_mode=pretraining_mode,
            )


def train(
    data_dir: Union[str, Path],
    model_checkpoint_dir: Union[str, Path] = "checkpoints",
    epochs: Optional[int] = None,
    steps: Optional[int] = 250000,
    input_size: Union[int, Sequence[int]] = 96,
    batch_size: int = 10,
    num_workers: int = 32,
    learning_rate: float = 1e-4,
    accumulate_grad_batches: int = 3,
    experiment_name: str = "default_experiment",
    resume_from_checkpoint: Optional[Union[str, Path]] = None,
    pretraining_mode: Literal["mae_only", "contrastive_only", "combined"] = "contrastive_only",
    contrastive_mode: Literal["regular", "modality_pairs"] = "modality_pairs",
    mae_batch_size: Optional[int] = None,
    num_checkpoints: int = 20,
    fast_dev_run: Union[bool, int] = False,
    seed: int = 42,
) -> None:
    """
    Args:
        data_dir:                 Directory containing training data
        model_checkpoint_dir:     Directory to save model checkpoints
        epochs:                   Maximum number of epochs to train
        steps:                    Maximum number of steps to train
        input_size:               Input image dimensions
        batch_size:               Batch size for training
        num_workers:              Number of workers for data loading
        learning_rate:            Maximum learning rate for training
        accumulate_grad_batches:  Number of gradient accumulation steps
        experiment_name:          Name of the experiment
        resume_from_checkpoint:   Optional path to checkpoint to resume from
        pretraining_mode:         Mode to use for pretraining (mae_only, contrastive_only, combined)
        contrastive_mode:         Mode to use for contrastive learning (regular, modality_pairs)
        mae_batch_size:           Optional separate batch size for MAE in combined mode
        num_checkpoints:          Number of intermediate checkpoints to save per training run
        fast_dev_run:             Quick debugging run
        seed:                     Random seed for reproducibility
    """
    save_dir: Union[str, Path] = Path(model_checkpoint_dir) / experiment_name

    # Early failure
    if epochs is None and steps is None:
        raise ValueError("Either `max_epochs` or `max_steps` must be provided")
    
    max_epochs = epochs if (steps is None) else None
    max_steps = steps or -1
    
    # Set random seed
    pl.seed_everything(seed)

    # Define transforms based on pretraining mode
    # MAE-only transforms (single volume key)
    mae_train_transforms = get_mae_transforms(
        keys=("volume",),
        input_size=input_size,
        val_mode=False,
    )
    mae_val_transforms = get_mae_transforms(
        keys=("volume",),
        input_size=input_size,
        val_mode=True,
    )

    # Contrastive transforms (vol1, vol2 keys) - used for contrastive_only and combined
    contrastive_train_transforms = get_contrastive_transforms(
        keys=("vol1", "vol2"),
        input_size=input_size,
        conservative_mode=True,
        val_mode=False,
        recon=pretraining_mode == "combined",
    )
    contrastive_val_transforms = get_contrastive_transforms(
        keys=("vol1", "vol2"),
        input_size=input_size,
        conservative_mode=True,
        val_mode=True,
        recon=pretraining_mode == "combined",
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath=save_dir, 
        filename="{epoch:02d}-{step}",
        every_n_train_steps=max_steps // num_checkpoints if max_steps != -1 else None,
        every_n_epochs=max_epochs // num_checkpoints if max_epochs is not None else None,
        save_last=True,
        save_top_k=-1,
    )

    # Create data module based on pretraining mode
    print("Creating data modules...")
    data_module = _create_data_module(
        pretraining_mode=pretraining_mode,
        data_dir=data_dir,
        mae_train_transforms=mae_train_transforms,
        mae_val_transforms=mae_val_transforms,
        contrastive_train_transforms=contrastive_train_transforms,
        contrastive_val_transforms=contrastive_val_transforms,
        contrastive_mode=contrastive_mode,
        batch_size=batch_size,
        num_workers=num_workers,
        mae_batch_size=mae_batch_size,
        input_size=input_size,
        seed=seed,
    )

    # Create or load model based on pretraining mode
    print("Creating model...")
    model = _create_or_load_model(
        pretraining_mode=pretraining_mode,
        resume_from_checkpoint=resume_from_checkpoint,
        input_size=input_size,
        learning_rate=learning_rate,
    )

    wandb_logger = WandbLogger(
        project="PretrainingFOMO25New",
        name=experiment_name,
        entity="matejgazda-technical-university-of-kosice",
        log_model=True,
    )

    print("Starting training...")
    import torch
    torch.set_float32_matmul_precision("high")
    trainer = pl.Trainer(
        accelerator="gpu",
        logger=wandb_logger,
        callbacks=[checkpoint_callback, LogLR(), LogGradNorm()],
        precision="bf16-mixed",
        accumulate_grad_batches=accumulate_grad_batches,
        max_epochs=max_epochs,
        max_steps=max_steps,
        log_every_n_steps=100,
        gradient_clip_val=10,
        gradient_clip_algorithm="norm",
        strategy=DDPStrategy(find_unused_parameters=True),
        fast_dev_run=fast_dev_run,
    )

    if resume_from_checkpoint:
        trainer.fit(model, datamodule=data_module, ckpt_path=resume_from_checkpoint)
    else:
        trainer.fit(model, datamodule=data_module)

    print("Training workflow completed.")


if __name__ == "__main__":
    Fire(train)
