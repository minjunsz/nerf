"""Core module implementing NeRF models."""
import itertools
from pathlib import Path
from typing import Literal

import lightning.pytorch as pl
import numpy as np
import torch
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.utilities.types import (
    EVAL_DATALOADERS,
    STEP_OUTPUT,
    TRAIN_DATALOADERS,
)
from torch import nn
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torchmetrics.functional import mean_squared_error

from src.dataset.lego import LegoDataset
from src.models.encoding.embedder import Embedder
from src.models.encoding.sinusoidal_positional_encoding import SinusoidalEmbedder
from src.utils.config_parser import Config
from src.utils.exceptions import NoDataException
from src.utils.get_rays import get_rays
from src.utils.rendering import render_rays


class MLPLayers(nn.Module):
    """Model architecture follows default model in the NeRF paper.
    8 linear layers with one skip connection at 4th layer.
    """

    def __init__(
        self,
        xyz_channel: int,
        viewdir_channel: int,
        hidden_dim: int = 256,
    ) -> None:
        """
        Parameters
        ----------
        xyz_channel : int
            Dimension of xyz input embeddings
        viewdir_channel : int
            Dimension of viewing direction input embeddings
        hidden_dim : int, optional
            Width of hidden layers, by default 256
        """
        super().__init__()
        self.xyz_channel = xyz_channel
        self.viewdir_channel = viewdir_channel
        self.hidden_dim = hidden_dim

        self.MLP_1 = nn.Sequential(
            nn.Linear(self.xyz_channel, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
        )
        self.MLP_2 = nn.Sequential(
            nn.Linear(self.xyz_channel + self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
        )
        self.feature_linear = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.alpha_linear = nn.Linear(self.hidden_dim, 1)
        self.view_linear = nn.Linear(
            self.viewdir_channel + self.hidden_dim,
            self.hidden_dim // 2,
        )
        self.rgb_linear = nn.Linear(self.hidden_dim // 2, 3)

    def forward(self, x):
        inputs, views = torch.split(x, [self.xyz_channel, self.viewdir_channel], dim=-1)
        h1 = self.MLP_1(inputs)
        h2 = self.MLP_2(torch.cat([inputs, h1], dim=-1))

        alpha = self.alpha_linear(h2)
        feature = self.feature_linear(h2)
        h3 = self.view_linear(torch.cat([views, feature], dim=-1))
        rgb = self.rgb_linear(h3)

        return torch.cat([rgb, alpha], dim=-1)


class NeRF(pl.LightningModule):
    xyz_embedder: Embedder
    viewdir_embedder: Embedder

    def __init__(
        self,
        config: Config,
        xyz_max_freq: int = 10,
        viewdir_max_freq: int = 4,
        hidden_dim: int = 256,
        initial_lr: float = 1e-4,
        near: float = 2.0,
        far: float = 6.0,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        data_path = Path(config.data_path)
        self.initial_lr = initial_lr
        self.near, self.far = near, far

        self.xyz_embedder = SinusoidalEmbedder(
            input_dim=3,
            num_freqs=xyz_max_freq + 1,
            max_freq=xyz_max_freq,
        )
        self.viewdir_embedder = SinusoidalEmbedder(
            input_dim=3,
            num_freqs=viewdir_max_freq + 1,
            max_freq=viewdir_max_freq,
        )
        self.coarse_model = MLPLayers(
            xyz_channel=self.xyz_embedder.out_dim,
            viewdir_channel=self.viewdir_embedder.out_dim,
            hidden_dim=hidden_dim,
        )
        self.fine_model = MLPLayers(
            xyz_channel=self.xyz_embedder.out_dim,
            viewdir_channel=self.viewdir_embedder.out_dim,
            hidden_dim=hidden_dim,
        )

        if not data_path.exists():
            raise NoDataException(f"No such data directory : {data_path}")
        self.train_dataset = LegoDataset(data_path, "train", half_res=True)
        self.val_dataset = LegoDataset(data_path, "val", half_res=True)
        self.test_dataset = LegoDataset(data_path, "test", half_res=True)

        self.batch_size = config.batch_size
        self.pre_crop_iters = config.pre_crop_iters
        self.pre_crop_frac = config.pre_crop_frac

        self.rendered_pixels: list[torch.Tensor] = []

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        if self.current_epoch < self.pre_crop_iters:
            return self.train_dataset.get_iterable_rays(
                pre_crop=True,
                pre_crop_frac=self.pre_crop_frac,
                batch_size=self.batch_size,
            )
        return self.train_dataset.get_iterable_rays(
            pre_crop=False, batch_size=self.batch_size
        )

    def on_train_epoch_start(self) -> None:
        print(f"start epoch {self.current_epoch}")

    def training_step(
        self, batch: tuple[torch.Tensor, ...], batch_idx: int
    ) -> STEP_OUTPUT:
        """
        batch: tuple of three tensors (rays_o, rays_d, gt_pixels).
            Each tensor has shape [BATCH_SIZE, 3]
        batch_idx: int
        """
        rays_o, rays_d, gt_pixels = batch

        rendered_result = render_rays(
            self,
            rays_o,
            rays_d,
            near=self.near,
            far=self.far,
        )

        coarse_loss = mean_squared_error(rendered_result["coarse"].rgb_map, gt_pixels)
        fine_loss = mean_squared_error(rendered_result["fine"].rgb_map, gt_pixels)
        total_loss = coarse_loss + fine_loss
        self.log_dict(
            {
                "train/loss": total_loss.detach(),
                "train/coarse_loss": coarse_loss.detach(),
                "train/fine_loss": fine_loss.detach(),
            }
        )
        return total_loss

    def configure_optimizers(self):
        params = itertools.chain(
            self.coarse_model.parameters(),
            self.fine_model.parameters(),
        )
        optimizer = torch.optim.Adam(
            params=params,
            lr=self.initial_lr,
            betas=(0.9, 0.999),
        )
        lr_scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)

        return [optimizer], [lr_scheduler]

    def val_dataloader(self) -> EVAL_DATALOADERS:
        return self.val_dataset.get_iterable_rays(
            pre_crop=False, batch_size=self.batch_size, randomize=False
        )

    def validation_step(
        self, batch: tuple[torch.Tensor, ...], batch_idx: int
    ) -> STEP_OUTPUT:
        """
        batch: tuple of three tensors (rays_o, rays_d, gt_pixels).
            Each tensor has shape [BATCH_SIZE, 3]
        batch_idx: int
        """
        rays_o, rays_d, gt_pixels = batch

        rendered_result = render_rays(
            self,
            rays_o,
            rays_d,
            near=self.near,
            far=self.far,
        )
        self.rendered_pixels.append(rendered_result["fine"].rgb_map.cpu())

        coarse_loss = mean_squared_error(rendered_result["coarse"].rgb_map, gt_pixels)
        fine_loss = mean_squared_error(rendered_result["fine"].rgb_map, gt_pixels)
        total_loss = coarse_loss + fine_loss
        self.log_dict(
            {
                "val/loss": total_loss.detach(),
                "val/coarse_loss": coarse_loss.detach(),
                "val/fine_loss": fine_loss.detach(),
            }
        )
        return total_loss.detach()

    def on_validation_epoch_end(self) -> None:
        img = (
            torch.cat(self.rendered_pixels, dim=0)
            .reshape(self.val_dataset.height, self.val_dataset.width, 3)
            .cpu()
            .numpy()
        )
        img = (img * 255.0).astype(np.uint8)
        wandb_logger: WandbLogger = self.logger  # type: ignore
        wandb_logger.log_image(
            key="samples",
            images=[self.val_dataset.current_image, img],
            caption=["GT", "Generated"],
        )
        self.rendered_pixels.clear()

    def synthesis_novel_view(self, height, width, K, c2w, chunk_size: int = 2048):
        """Render a novel view with given parameters K, c2W

        Parameters
        ----------
        height : int
            Height of the image
        width : int
            Width of the image
        K : torch.Tensor
            A transformation matrix from normalized image plane to pixel coordinate. This matrix consists of intrinsic parameters of the camera.
        c2w : torch.Tensor
            A transformation matrix from camera coordinate to world coordinate. This matrix adopts column-major manner.
        chunk_size : int, optional
            Chunk size to compute at once. Too large chunk size gives GPU out-of-memory, by default 2048

        Returns
        -------
        numpy.ndarray
            Image in numpy array. [height, width, 3]
        """
        rays_o, rays_d = get_rays(height, width, K, c2w=c2w)
        rays_o, rays_d = rays_o.flatten(0, 1), rays_d.flatten(0, 1)

        rendered_rays: list[torch.Tensor] = []

        for i in range(0, height * width, chunk_size):
            rays_o_batch = rays_o[i : i + chunk_size]
            rays_d_batch = rays_d[i : i + chunk_size]
            rendered_result = render_rays(
                self,
                rays_o_batch,
                rays_d_batch,
                near=self.near,
                far=self.far,
            )
            rendered_rays.append(rendered_result["fine"].rgb_map.cpu())

        img = torch.cat(rendered_rays, dim=0).reshape(height, width, 3).cpu().numpy()
        img: np.ndarray = (img * 255.0).astype(np.uint8)
        return img

    def estimate_RGBd(
        self,
        xyz_points: torch.Tensor,
        viewdir: torch.Tensor,
        network: Literal["coarse", "fine"],
    ) -> torch.Tensor:
        """Return RGBd values for the given xyz points.

        Parameters
        ----------
        xyz_points : torch.Tensor
            input points' 3D coordinate in World Coordinate system. [N_rays, N_samples,3]
        viewdir : torch.Tensor
            input data's viewing direction vector in World Coordiante system. Shape [N_rays,3]

        Returns
        -------
        torch.Tensor
            Estimated RGBd values for each point [N_rays, N_samples, 3]
        """
        xyz_flat = xyz_points.flatten(start_dim=0, end_dim=-2)
        xyz_embedding = self.xyz_embedder.embed(xyz_flat)  # [N_rays*N_samples,3]

        viewdir_expand = viewdir[..., None, :].expand_as(xyz_points)
        viewdir_flat = viewdir_expand.flatten(start_dim=0, end_dim=-2)
        viewdir_embedding = self.viewdir_embedder.embed(viewdir_flat)

        embedding = torch.cat([xyz_embedding, viewdir_embedding], dim=-1)
        model = self.coarse_model if network == "coarse" else self.fine_model
        output_flat = model(embedding)
        output = output_flat.reshape(xyz_points.shape[:-1] + output_flat.shape[-1:])
        return output

    def estimate_density(self, xyz_points: torch.Tensor) -> torch.Tensor:
        model = self.fine_model
        dummy_viewdir = torch.empty_like(xyz_points)
        xyz_embedding = self.xyz_embedder.embed(xyz_points)
        viewdir_embedding = self.viewdir_embedder.embed(dummy_viewdir)
        embedding = torch.cat([xyz_embedding, viewdir_embedding], dim=-1)
        output = model(embedding)
        alpha = output[:, -1]
        return alpha
