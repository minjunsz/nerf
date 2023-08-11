"""This module includes whole training pipeline for NeRF."""
from pathlib import Path

import numpy as np
import torch

from src.dataset.lego import LegoDataset, read_camera_params
from src.models.nerf import Embedder, NeRF
from src.utils.config_parser import get_config
from src.utils.exceptions import NoDataException
from src.utils.get_rays import get_rays


def instantiate_model(
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
    xyz_embedder = Embedder(
        input_dim=3,
        num_freqs=xyz_max_freq + 1,
        max_freq=xyz_max_freq,
    )
    viewdir_embedder = Embedder(
        input_dim=3,
        num_freqs=viewdir_max_freq + 1,
        max_freq=viewdir_max_freq,
    )
    coarse_model = NeRF(
        xyz_channel=xyz_embedder.out_dim,
        viewdir_channel=viewdir_embedder.out_dim,
        hidden_dim=hidden_dim,
    )

    fine_model = NeRF(
        xyz_channel=xyz_embedder.out_dim,
        viewdir_channel=viewdir_embedder.out_dim,
        hidden_dim=hidden_dim,
    )

    return coarse_model, fine_model, xyz_embedder, viewdir_embedder


def train():
    config = get_config()
    data_path = Path(config.data_path)
    if not data_path.exists():
        raise NoDataException(f"No such data directory : {data_path}")
    train_dataset = LegoDataset(data_path, "train", half_res=True)
    val_dataset = LegoDataset(data_path, "val", half_res=True)
    test_dataset = LegoDataset(data_path, "test", half_res=True)

    near, far = 2.0, 6.0
    height, width, focal_length = read_camera_params(data_path, "train", half_res=True)
    K = torch.tensor(
        [
            [focal_length, 0, 0.5 * width],
            [0, focal_length, 0.5 * height],
            [0, 0, 1],
        ]
    )

    coarse_model, fine_model, xyz_embedder, viewdir_embedder = instantiate_model()
    for iteration in range(config.train_iters):
        img_idx = np.random.randint(len(train_dataset))
        data = train_dataset[img_idx]
        img, pose = data.imgs, data.poses[:3, :4]
        rays_o, rays_d = get_rays(height, width, K, c2w=pose)

        if iteration < config.pre_crop_iters:
            dH = int(height // 2 * config.pre_crop_frac)
            dW = int(width // 2 * config.pre_crop_frac)
            coords = torch.stack(
                torch.meshgrid(
                    torch.linspace(height // 2 - dH, height // 2 + dH - 1, 2 * dH),
                    torch.linspace(width // 2 - dW, width // 2 + dW - 1, 2 * dW),
                    indexing="ij",
                ),
                dim=-1,
            )
        else:
            coords = torch.stack(
                torch.meshgrid(
                    torch.linspace(0, height - 1, height),
                    torch.linspace(0, width - 1, width),
                    indexing="ij",
                ),
                dim=-1,
            )  # (H, W, 2)

        coords = torch.reshape(coords, [-1, 2])  # (H * W, 2)
        select_inds = np.random.choice(
            coords.shape[0],
            size=[config.N_rand],
            replace=False,
        )  # (N_rand,)
        select_coords = coords[select_inds].long()  # (N_rand, 2)
        rays_o = rays_o[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)
        rays_d = rays_d[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)
        target_pixels = img[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)


if __name__ == "__main__":
    train()
