"""This module includes whole training pipeline for NeRF."""
import lightning.pytorch as pl
import torch
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint, Timer
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.profilers import PyTorchProfiler

from src.models.nerf_pl import NeRF
from src.utils.config_parser import get_config


def train():
    config = get_config()
    pl.seed_everything(100)

    if torch.cuda.is_available():
        torch.set_default_device("cuda:1")

    nerf = NeRF(config)
    wandb_logger = WandbLogger(project="nerf", log_model="all")
    checkpoint_callback = ModelCheckpoint(monitor="val/loss", mode="min")
    lr_monitor = LearningRateMonitor(logging_interval="epoch")
    timer = Timer(duration="00:12:00:00")
    profiler = PyTorchProfiler(dirpath='.', filename='pytorch_profile')
    trainer = pl.Trainer(
        logger=wandb_logger,
        max_epochs=config.train_iters,
        check_val_every_n_epoch=5,
        callbacks=[checkpoint_callback, lr_monitor, timer],
        accelerator="gpu",
        devices=[1],
        num_sanity_val_steps=0,
        reload_dataloaders_every_n_epochs=1,
        profiler=profiler,
    )
    wandb_logger.watch(nerf, log="all")
    trainer.fit(model=nerf)


if __name__ == "__main__":
    train()
