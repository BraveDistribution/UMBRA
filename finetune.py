from pathlib import Path
from typing import Optional, Union, Optional, Literal, List

import lightning.pytorch as pl
from fire import Fire
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
import wandb
from lightning.pytorch.strategies import DDPStrategy
import torch

from data.finetuning_datamodule import FinetuningDataModule
from models.finetuning import SegmentationSwinFPN
from models.callbacks import LogLR, LogGradNorm
from transforms.composed import get_segmentation_transforms


def train(
    data_dir: Union[str, Path],
    labels_dir: Optional[Union[str, Path]] = None,
    finetuning_task: Literal["segmentation"] = "segmentation",
    modalities: Optional[Union[str, List[str], List[List[str]]]] = "t1",
    scan_type: Literal["numpy", "nifti"] = "numpy",
    target: Literal["label", "mask", "combined"] = "mask",
    test_dir: Optional[Union[str, Path]] = None,
    test_labels_dir: Optional[Union[str, Path]] = None,
    subset_train: Optional[Union[List[float], float]] = None,
    model_checkpoint_dir: Union[str, Path] = "checkpoints",
    load_encoder_from: Optional[Union[str, Path]] = None,
    unfreeze_encoder_at: Union[int, float] = 0.3,
    epochs: Optional[int] = None,
    steps: Optional[int] = 20000,
    input_size: Union[int, List[int]] = 96,
    batch_size: int = 8,
    num_workers: int = 8,
    learning_rate: float = 5e-4,
    encoder_lr_ratio: float = 0.1,
    accumulate_grad_batches: int = 1,
    experiment_name: str = "finetuning",
    resume_from_checkpoint: Optional[Union[str, Path]] = None,
    num_checkpoints: int = 1,
    fast_dev_run: Union[bool, int] = False,
    do_test: bool = True,
    seed: int = 42,
    **overrides
) -> None:
    """
    Args:
        data_dir:                 Directory containing training data
        labels_dir:               Directory containing labels; otherwise same as `data_dir`.
        finetuning_task:          Task to finetune; only segmentation for now.
        modalities:               Modalities to use; will iterate across each set if 
                                    provided as a nested sequence.
        scan_type:                Type of the scan; `numpy` or `nifti`
        target:                   Target to finetune; `label`, `mask`, or `combined`
        test_dir:                 Directory containing test data; otherwise, train/test split is used.
        test_labels_dir:          Directory containing test labels; otherwise same as `test_dir`.
        subset_train:             Subset of the train data to use; will iterate across each set if 
                                    provided as a sequence.
        model_checkpoint_dir:     Directory to save model checkpoints
        load_encoder_from:        Optional path to checkpoint to load encoder from
        epochs:                   Maximum number of epochs to train
        steps:                    Maximum number of steps to train
        input_size:               Input image dimensions
        batch_size:               Batch size for training
        num_workers:              Number of workers for data loading
        learning_rate:            Maximum learning rate for training
        accumulate_grad_batches:  Number of gradient accumulation steps
        experiment_name:          Name of the experiment
        resume_from_checkpoint:   Optional path to checkpoint to resume from
        load_encoder_from:        Optional path to checkpoint to load encoder from
        num_checkpoints:          Number of intermediate checkpoints to save per training run
        fast_dev_run:             Quick debugging run
        seed:                     Random seed for reproducibility
        **overrides:              Additional keyword arguments for `pl.Trainer`
    """
    # Early failure
    if finetuning_task not in ["segmentation"]:
        raise ValueError(f"Invalid finetuning task: {finetuning_task}")

    if epochs is None and steps is None:
        raise ValueError("Either `max_epochs` or `max_steps` must be provided")

    if overrides:
        print("[Fire kwargs] extra overrides for `pl.Trainer`:", overrides)
    
    max_epochs = epochs if (steps is None) else None
    max_steps = steps or -1

    # Set up modalities loop
    mod_loop: List[List[str]]
    if isinstance(modalities, type(None)):
        mod_loop = [["volume"]]
    elif isinstance(modalities, str):
        mod_loop = [[modalities]]
    elif isinstance(modalities, list):
        mod_loop = [list(mod) for mod in modalities]
    else:
        raise ValueError("`modalities` must be a string or a sequence of strings")
    
    # Set up subset train loop
    few_shot_loop: List[Optional[float]]
    if isinstance(subset_train, type(None)):
        few_shot_loop = [None]
    elif isinstance(subset_train, float):
        few_shot_loop = [subset_train]
    elif isinstance(subset_train, list):
        few_shot_loop = [s for s in subset_train]
    else:
        raise ValueError("`subset_train` must be a float or a sequence of floats")

    # Outer loop: modalities
    for mod in mod_loop:
        # Inner loop: few-shot
        for few_shot in few_shot_loop:
            # Configure experiment
            if mod is not None:
                mod_str = ", ".join(mod)
            else:
                mod_str = "all"
            if few_shot is not None:
                few_shot_str = f"fewshot-{int(few_shot*100)}%"
            else:
                few_shot_str = "full"
            experiment_name = f"{finetuning_task}_{mod_str}_{few_shot_str}"

            save_dir: Union[str, Path] = Path(model_checkpoint_dir) / experiment_name

            wandb_logger = WandbLogger(
                project="FinetuningFOMO25",
                name=experiment_name,
                entity="matejgazda-technical-university-of-kosice",
                log_model=True,
            )
    
            # Set random seed
            pl.seed_everything(seed, workers=True)

            # Define transforms based on finetuning task
            if finetuning_task == "segmentation":
                train_transforms = get_segmentation_transforms(
                    input_size=input_size,
                    keys=mod,
                    seg_key="mask",
                    out_key="volume",
                    n_patches=4,
                    val_mode=False,
                )
                val_transforms = get_segmentation_transforms(
                    input_size=input_size,
                    keys=mod,
                    seg_key="mask",
                    out_key="volume",
                    n_patches=4,
                    val_mode=True,
                )
            
                data_module = FinetuningDataModule(
                    data_dir=data_dir,
                    labels_dir=labels_dir,
                    train_transforms=train_transforms,
                    val_transforms=val_transforms,
                    modalities=mod, 
                    scan_type=scan_type,
                    target=target,
                    require_all_labels=True,
                    require_all_scans=True,
                    test_dir=test_dir,
                    test_labels_dir=test_labels_dir,
                    train_val_split=0.2,
                    train_test_split=0.2,
                    subset_train=few_shot,
                    batch_size=batch_size,
                    num_workers=num_workers,
                    seed=seed,
                )

                model = SegmentationSwinFPN(
                    image_size=input_size,
                    learning_rate=learning_rate,
                    warmup=0.05,
                    load_encoder_from=load_encoder_from,
                    encoder_prefix_in_ckpt='model.encoder',
                    unfreeze_encoder_at=unfreeze_encoder_at,
                    encoder_lr_ratio=encoder_lr_ratio,
                    input_key="volume",
                    target_key="mask",
                )
            else:
                raise ValueError(f"Invalid finetuning task: {finetuning_task}")

            checkpoint_callback = ModelCheckpoint(
                dirpath=save_dir, 
                filename="{epoch:02d}-{step}",
                every_n_train_steps=max_steps // num_checkpoints if max_steps != -1 else None,
                every_n_epochs=max_epochs // num_checkpoints if max_epochs is not None else None,
                save_last=True,
                save_top_k=-1,
            )

            # Set float32 matmul precision to high for better performance
            torch.set_float32_matmul_precision("medium")

            print("Starting training...")
            trainer = pl.Trainer(
                accelerator="gpu",
                logger=wandb_logger,
                callbacks=[checkpoint_callback, LogLR(), LogGradNorm()],
                precision="bf16-mixed",
                accumulate_grad_batches=accumulate_grad_batches,
                max_epochs=max_epochs,
                max_steps=max_steps,
                log_every_n_steps=10,
                gradient_clip_val=2,
                gradient_clip_algorithm="norm",
                strategy=DDPStrategy(find_unused_parameters=True),
                fast_dev_run=fast_dev_run,
                **overrides,
            )

            if resume_from_checkpoint:
                trainer.fit(model, datamodule=data_module, ckpt_path=resume_from_checkpoint)
            else:
                trainer.fit(model, datamodule=data_module)

            if do_test:
                trainer.test(model, datamodule=data_module)

            print(f"Finetuning completed for combination: modalities={mod}, few_shot={few_shot}.")

            wandb.finish()
        
    print("All finetuning combinations completed.")


if __name__ == "__main__":
    Fire(train)
