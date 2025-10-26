from pathlib import Path
from typing import Optional, Union, Literal, List

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


def train_and_evaluate_run(
    data_dir: Union[str, Path],
    finetuning_task: Literal["segmentation"],
    experiment_name: str,
    modalities: List[str],
    test_dir: Union[str, Path],
    target: Literal["label", "mask", "combined"] = "mask",
    subset_train: Optional[float] = None,
    model_checkpoint_dir: Union[str, Path] = "checkpoints",
    load_encoder_from: Optional[Union[str, Path]] = None,
    unfreeze_encoder_at: Union[int, float] = 0.3,
    epochs: Optional[int] = None,
    steps: int = 20000,
    input_size: Union[int, List[int]] = 96,
    scan_type: Literal["numpy", "nifti"] = "numpy",
    batch_size: int = 8,
    num_workers: int = 8,
    learning_rate: float = 5e-4,
    encoder_lr_ratio: float = 0.1,
    weight_decay: float = 1e-2,
    encoder_wd: float = 1e-3,
    accumulate_grad_batches: int = 1,
    resume_from_checkpoint: Optional[Union[str, Path]] = None,
    num_checkpoints: int = 1,
    fast_dev_run: Union[bool, int] = False,
    do_test: bool = True,
    seed: int = 42,
    **overrides
) -> None:
    """
    Train and evaluate a model on a segmentation task for a single experiment,
    defined by:
    - Set of modalities
    - Subset of the train data to use
    - Pretrained checkpoint (if provided)

    Args:
        data_dir:                 Directory containing training data
        finetuning_task:          Task to finetune; only segmentation for now.
        experiment_name:          Name of the experiment
        modalities:               Modalities to use; will iterate across each set if 
                                    provided as a nested sequence.
        test_dir:                 Directory containing test data; otherwise, train/test split is used.
        target:                   Target to finetune; `label`, `mask`, or `combined`
        subset_train:             Subset of the train data to use; will iterate across each set if 
                                    provided as a sequence.
        model_checkpoint_dir:     Directory to save model checkpoints
        load_encoder_from:        Optional path to checkpoint to load encoder from
        unfreeze_encoder_at:      Step or fraction of total steps at which to unfreeze encoder.
        epochs:                   Maximum number of epochs to train
        steps:                    Maximum number of steps to train
        input_size:               Input image dimensions
        scan_type:                Type of the scan; `numpy` or `nifti`
        batch_size:               Batch size for training
        num_workers:              Number of workers for data loading
        learning_rate:            Maximum learning rate for all parameters except encoder
        encoder_lr_ratio:         Learning rate ratio for encoder relative to `learning_rate`
        weight_decay:             Weight decay for all parameters except encoder
        encoder_wd:               Weight decay for encoder parameters
        accumulate_grad_batches:  Number of gradient accumulation steps
        resume_from_checkpoint:   Optional path to checkpoint to resume from
        num_checkpoints:          Number of intermediate checkpoints to save per training run
        fast_dev_run:             Quick debugging run
        seed:                     Random seed for reproducibility
        **overrides:              Additional keyword arguments for `pl.Trainer`
    """
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
            keys=modalities,
            seg_key="mask",
            out_key="volume",
            n_patches=4,
            val_mode=False,
        )
        val_transforms = get_segmentation_transforms(
            input_size=input_size,
            keys=modalities,
            seg_key="mask",
            out_key="volume",
            n_patches=4,
            val_mode=True,
        )
    
        data_module = FinetuningDataModule(
            data_dir=data_dir,
            train_transforms=train_transforms,
            val_transforms=val_transforms,
            modalities=modalities, 
            scan_type=scan_type,
            target=target,
            require_all_labels=True,
            require_all_scans=True,
            test_dir=test_dir,
            train_val_split=0.2,
            subset_train=subset_train,
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
            wd_encoder=encoder_wd,
            wd_rest=weight_decay,
            input_key="volume",
            target_key="mask",
        )
    else:
        raise ValueError(f"Invalid finetuning task: {finetuning_task}")

    checkpoint_callback = ModelCheckpoint(
        dirpath=model_checkpoint_dir, 
        filename="{epoch:02d}-{step}",
        every_n_train_steps=steps // num_checkpoints if steps != -1 else None,
        every_n_epochs=epochs // num_checkpoints if epochs is not None else None,
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
        max_epochs=epochs,
        max_steps=steps,
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

    print(f"Finetuning completed for experiment: {experiment_name}.")

    wandb.finish()


def experiment_loop(
    # Required; common to all experiments
    data_dir: Union[str, Path],
    test_dir: Union[str, Path],
    experiment_name: str,
    # Define loops; in the following hierarchy
    modalities: Union[str, List[str], List[List[str]]],
    subset_train: Optional[Union[List[float], float]] = None,
    encoder_ckpts: Optional[Union[str, Path, List[Optional[Union[str, Path]]]]] = None,
    # Unique per encoder checkpoint; repeated across modalities/few-shot subsets
    name: Optional[Union[str, List[str]]] = None, # logging-only
    unfreeze_encoder_at: Union[List[Union[int, float]], Union[int, float]] = 0.3,
    encoder_lr_ratio: Union[List[float], float] = 0.1,
    encoder_wd: Union[List[float], float] = 1e-3,
    resume_from_checkpoint: Optional[List[Optional[Union[str, Path]]]] = None,
    # Defaults; repeated across experiments
    epochs: Optional[int] = None,
    steps: Optional[int] = 20000,
    num_workers: int = 8,
    batch_size: int = 8,
    accumulate_grad_batches: int = 1,
    learning_rate: float = 5e-4,
    weight_decay: float = 1e-2,
    input_size: Union[List[int], int] = 96,
    finetuning_task: Literal["segmentation"] = "segmentation",
    scan_type: Literal["numpy", "nifti"] = "numpy",
    target: Literal["label", "mask", "combined"] = "mask",
    model_checkpoint_dir: Union[str, Path] = "checkpoints",
    num_checkpoints: int = 1,
    fast_dev_run: Union[bool, int] = False,
    do_test: bool = True,
    seed: int = 42,
    **overrides
) -> None:
    """
    Run experiments on a given dataset, looping over the following:
    - Modalities
    - Subset of the train data
    - Pretrained checkpoints (if provided)

    Args:
        data_dir:                 Directory containing training data
        test_dir:                 Directory containing test data
        experiment_name:          Name of the experiment
        modalities:               Modalities to use; will iterate across each set if 
                                    provided as a nested sequence.
        subset_train:             Subset of the train data to use; will iterate across each set if 
                                    provided as a sequence.
        encoder_ckpts:            Optional path(s) to checkpoint(s) to load encoder from
        name:                     Name of the encoder to use, instead of the full path 
                                    when loading from a checkpoint.
        unfreeze_encoder_at:      Step or fraction of total steps at which to unfreeze encoder
        learning_rate:            Maximum learning rate for training
        encoder_lr_ratio:         Learning rate ratio for encoder relative to decoder
        resume_from_checkpoint:   Optional path(s) to checkpoint(s) to resume from
        epochs:                   Maximum number of epochs to train
        steps:                    Maximum number of steps to train
        num_workers:              Number of workers for data loading
        batch_size:               Batch size for training
        accumulate_grad_batches:  Number of gradient accumulation steps
        input_size:               Input image dimensions
        finetuning_task:          Task to finetune; only segmentation for now
        scan_type:                Type of the scan; `numpy` or `nifti`
        target:                   Target to finetune; `label`, `mask`, or `combined`
        model_checkpoint_dir:     Directory to save model checkpoints
        num_checkpoints:          Number of intermediate checkpoints to save per training run
        fast_dev_run:             Quick debugging run
        do_test:                  Whether to run test evaluation after training
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
    
    # Prioritize steps
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

    # Set up encoder checkpoint loop
    encoder_ckpt_loop: List[Optional[Union[str, Path]]]
    if isinstance(encoder_ckpts, type(None)):
        encoder_ckpt_loop = [None]
    elif isinstance(encoder_ckpts, str):
        encoder_ckpt_loop = [encoder_ckpts]
    elif isinstance(encoder_ckpts, list):
        encoder_ckpt_loop = [e for e in encoder_ckpts]
    else:
        raise ValueError("`encoder_ckpts` must be a string or a sequence of strings")

    # Standardize loop-specific entries:
    # ckpt name alias for logging
    if isinstance(name, list):
        if len(name) != len(encoder_ckpt_loop):
            raise ValueError("`name` and `encoder_ckpts` must have the same length")
        name_loop = [str(n) for n in name]
    elif isinstance(name, str):
        name_loop = [name] * len(encoder_ckpt_loop)
    elif isinstance(name, type(None)):
        name_loop = [None] * len(encoder_ckpt_loop)
    else:
        raise ValueError("`name` must be a string or a sequence of strings")
    
    # encoder unfreezing
    if isinstance(unfreeze_encoder_at, list):
        if len(unfreeze_encoder_at) != len(encoder_ckpt_loop):
            raise ValueError(
                "`unfreeze_encoder_at` and `encoder_ckpts` must have the same length"
            )
        for i in unfreeze_encoder_at:
            if not isinstance(i, (int, float)):
                raise ValueError(
                    "`unfreeze_encoder_at` must be an integer/float or a sequence of integers/floats"
                )
    elif isinstance(unfreeze_encoder_at, (int, float)):
        unfreeze_encoder_at_loop = [unfreeze_encoder_at] * len(encoder_ckpt_loop)
    else:
        raise ValueError(
            "`unfreeze_encoder_at` must be an integer/float or a sequence of integers/floats"
        )

    # encoder learning rate ratio
    if isinstance(encoder_lr_ratio, list):
        if len(encoder_lr_ratio) != len(encoder_ckpt_loop):
            raise ValueError("`encoder_lr_ratio` and `encoder_ckpts` must have the same length")
        encoder_lr_ratio_loop = [float(e) for e in encoder_lr_ratio]
    elif isinstance(encoder_lr_ratio, float):
        encoder_lr_ratio_loop = [encoder_lr_ratio] * len(encoder_ckpt_loop)
    else:
        raise ValueError("`encoder_lr_ratio` must be a float or a sequence of floats")

    # encoder weight decay
    if isinstance(encoder_wd, list):
        if len(encoder_wd) != len(encoder_ckpt_loop):
            raise ValueError("`encoder_wd` and `encoder_ckpts` must have the same length")
        encoder_wd_loop = [float(e) for e in encoder_wd]
    elif isinstance(encoder_wd, float):
        encoder_wd_loop = [encoder_wd] * len(encoder_ckpt_loop)
    else:
        raise ValueError("`encoder_wd` must be a float or a sequence of floats")

    # resume from checkpoint
    if isinstance(resume_from_checkpoint, list):
        if len(resume_from_checkpoint) != len(encoder_ckpt_loop):
            raise ValueError("`resume_from_checkpoint` and `encoder_ckpts` must have the same length")
        resume_from_checkpoint_loop = [str(r) for r in resume_from_checkpoint]
    elif isinstance(resume_from_checkpoint, (str, Path)):
        resume_from_checkpoint_loop = [resume_from_checkpoint] * len(encoder_ckpt_loop)
    elif isinstance(resume_from_checkpoint, type(None)):
        resume_from_checkpoint_loop = [None] * len(encoder_ckpt_loop)
    else:
        raise ValueError("`resume_from_checkpoint` must be a string or a sequence of strings")

    # Loop over modalities
    for mod in mod_loop:
        # Loop over few-shot subsets
        for few_shot in few_shot_loop:
            # Loop over encoder checkpoints
            for i, load_encoder_from in enumerate(encoder_ckpt_loop):
                # Set up experiment name:
                # modalities
                if mod is not None:
                    mod_str = ", ".join(mod)
                else:
                    mod_str = "all"
                # few-shot
                if few_shot is not None:
                    few_shot_str = f"fewshot-{int(few_shot*100)}%"
                else:
                    few_shot_str = "full"
                # encoder checkpoint
                if name_loop[i] is not None:
                    name_str = name_loop[i]
                else:
                    name_str = str(i)
                
                experiment_name = f"{finetuning_task}_{mod_str}_{few_shot_str}_{name_str}"

                save_dir: Union[str, Path] = Path(model_checkpoint_dir) / experiment_name

                # Train and evaluate
                train_and_evaluate_run(
                    data_dir=data_dir,
                    finetuning_task=finetuning_task,
                    experiment_name=experiment_name,
                    modalities=mod,
                    test_dir=test_dir,
                    target=target,
                    subset_train=few_shot,
                    model_checkpoint_dir=save_dir,
                    load_encoder_from=load_encoder_from,
                    unfreeze_encoder_at=unfreeze_encoder_at_loop[i],
                    epochs=max_epochs,
                    steps=max_steps,
                    input_size=input_size,
                    scan_type=scan_type,
                    batch_size=batch_size,
                    num_workers=num_workers,
                    learning_rate=learning_rate,
                    encoder_lr_ratio=encoder_lr_ratio_loop[i],
                    weight_decay=weight_decay,
                    encoder_wd=encoder_wd_loop[i],
                    accumulate_grad_batches=accumulate_grad_batches,
                    resume_from_checkpoint=resume_from_checkpoint_loop[i],
                    num_checkpoints=num_checkpoints,
                    fast_dev_run=fast_dev_run,
                    do_test=do_test,
                    seed=seed,
                    **overrides,
                )
    
    print("All finetuning combinations completed.")


if __name__ == "__main__":
    Fire(experiment_loop)
