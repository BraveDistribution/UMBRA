try:
    import fire
except ImportError:
    fire = None  # type: ignore

try:
    import lightning as L
    from lightning.pytorch.loggers import WandbLogger
except ImportError:
    L = None  # type: ignore
    WandbLogger = None  # type: ignore

from configs.task_config import CONFIG


def train(*args, task_config=None, **kwargs):
    """
    Training function with parametric arguments.

    Args:
        *args: Positional arguments
        task_config: Path to task config file or config dict
        **kwargs: Keyword arguments
    """
    config = CONFIG(task_config=task_config)

    print(f"Training with args: {args}")
    print(f"Training with config: {config.__dict__}")
    print(f"Training with kwargs: {kwargs}")

    if WandbLogger is None or L is None:
        raise ImportError("lightning and wandb packages are required")

    wandb_logger = WandbLogger(project="UMBRA")

    trainer = L.Trainer(
        devices=config.devices,
        accelerator="gpu",
        max_epochs=config.max_epochs,
        logger=wandb_logger,
    )

    model: L.LightningModule = None  # type: ignore
    data_module: L.LightningDataModule = None  # type: ignore
    trainer.fit(model, datamodule=data_module)


if __name__ == "__main__":
    if fire is not None:
        fire.Fire(train)
    else:
        print("Error: fire package not installed")
