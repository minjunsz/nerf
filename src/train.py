"""This module includes whole training pipeline for NeRF."""
import time
from pathlib import Path

import lightning.pytorch as pl
import numpy as np
import torch
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from torchmetrics.functional import mean_squared_error

import wandb
from src.dataset.lego import LegoDataset
from src.models.encoding.sinusoidal_positional_encoding import SinusoidalEmbedder
from src.models.nerf_pl import NeRF
from src.utils.config_parser import Config, get_config
from src.utils.exceptions import NoDataException
from src.utils.get_rays import get_rays
from src.utils.rendering import render_rays


def instantiate_model(
    config: Config,
    xyz_max_freq: int = 10,
    viewdir_max_freq: int = 4,
    hidden_dim: int = 256,
):
    """Instantiate embedders and core NeRF models.
    Parameters
    ----------
    xyz_max_freq : int, optional
        log_2(MAX_FREQ) for xyz positional encoding, by default 10
    viewdir_max_freq : int, optional
        log_2(MAX_FREQ) for viewing direction encoding, by default 4
    hidden_dim : int, optional
        width of hidden layer, by default 256

    Returns
    -------
    tuple[NeRF, NeRF, Embedder, Embedder]
        Return tuple; (coarse_model, fine_model, xyz_embedder, viewdir_embedder)
    """
    data_path = Path(config.data_path)

    xyz_embedder = SinusoidalEmbedder(
        input_dim=3,
        num_freqs=xyz_max_freq + 1,
        max_freq=xyz_max_freq,
    )
    viewdir_embedder = SinusoidalEmbedder(
        input_dim=3,
        num_freqs=viewdir_max_freq + 1,
        max_freq=viewdir_max_freq,
    )
    nerf = NeRF(
        xyz_embedder,
        viewdir_embedder,
        data_path=data_path,
        hidden_dim=hidden_dim,
    )

    return nerf


def train():
    config = get_config()

    nerf = instantiate_model(config)

    wandb_logger = WandbLogger(project="nerf", log_model="all")
    checkpoint_callback = ModelCheckpoint(monitor="val/loss", mode="min")
    trainer = pl.Trainer(
        logger=wandb_logger,
        check_val_every_n_epoch=5,
        callbacks=[checkpoint_callback],
        accelerator="gpu",
        devices=[1],
    )
    trainer.fit(model=nerf)


if __name__ == "__main__":
    train()
