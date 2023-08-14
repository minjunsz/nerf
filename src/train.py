"""This module includes whole training pipeline for NeRF."""
import time
from pathlib import Path

import numpy as np
import torch
from torchmetrics.functional import mean_squared_error

from src.dataset.lego import LegoDataset, read_camera_params
from src.models.nerf import Embedder, NeRF
from src.utils.config_parser import get_config
from src.utils.exceptions import NoDataException
from src.utils.get_rays import get_rays
from src.utils.rendering import render_img


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
    if torch.cuda.is_available():
        torch.set_default_device("cuda:1")
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

    initial_lr, lr_decay_step = 5e-4, 20
    current_lr = initial_lr
    optimizer = torch.optim.Adam(
        params=list(coarse_model.parameters()) + list(fine_model.parameters()),
        lr=current_lr,
        betas=(0.9, 0.999),
    )
    loss_fn = mean_squared_error

    for iteration in range(config.train_iters):
        if iteration % config.log_interval == 0:
            print(f"Iteration {iteration} start : {time.ctime(time.time())}")

        img_idx = np.random.randint(len(train_dataset))
        print(f"Train with image {img_idx}")
        data = train_dataset[img_idx]
        img, pose = data.imgs, data.poses
        rays_o_whole, rays_d_whole = get_rays(height, width, K, c2w=pose)

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

        coords = coords.reshape(-1, 2)  # (H * W, 2)
        random_idx = np.random.permutation(coords.size(0))

        acc_loss = 0.0

        for i in range(0, coords.size(0), config.N_rand):
            random_idx_chunck = random_idx[i : i + config.N_rand]
            select_coords = coords[random_idx_chunck].long()
            rays_o = rays_o_whole[
                select_coords[:, 0], select_coords[:, 1]
            ]  # (N_rand, 3)
            rays_d = rays_d_whole[
                select_coords[:, 0], select_coords[:, 1]
            ]  # (N_rand, 3)
            target_pixels = img[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)

            rendered_result = render_img(
                coarse_model,
                fine_model,
                rays_o,
                rays_d,
                xyz_embedder,
                viewdir_embedder,
                near=near,
                far=far,
            )

            optimizer.zero_grad()
            img_loss = loss_fn(rendered_result["fine"].rgb_map, target_pixels)
            coarse_loss = loss_fn(rendered_result["coarse"].rgb_map, target_pixels)
            total_loss = img_loss + coarse_loss
            total_loss.backward()
            optimizer.step()
            acc_loss += total_loss.item()

        if iteration % lr_decay_step == lr_decay_step - 1:
            current_lr *= 0.1
            for param_group in optimizer.param_groups:
                param_group["lr"] = current_lr

        if iteration % config.log_interval == 0:
            print(f"Iteration {iteration}: loss {acc_loss}")
            PATH = (
                Path.cwd() / "checkpoints" / str(config.run_id) / f"iter_{iteration}.pt"
            )
            torch.save(
                {
                    "iteration": iteration,
                    "coarse_model": coarse_model.state_dict(),
                    "fine_model": fine_model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": acc_loss,
                },
                PATH,
            )


if __name__ == "__main__":
    train()
