"""This module includes whole training pipeline for NeRF."""
from typing import Callable
from dataclasses import dataclass

import torch
from torch import linalg as LA

from src.models.nerf import Embedder, NeRF
from src.utils.rendering import sample_coarse_points, integrate_ray


@dataclass
class ModelArgs:
    xyz_max_freq: int = 10  #
    viewdir_max_freq: int = 4  #
    hidden_dim: int = 256  #


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
    model = NeRF(
        xyz_channel=xyz_embedder.out_dim,
        viewdir_channel=viewdir_embedder.out_dim,
        hidden_dim=hidden_dim,
    )

    # fine_model = NeRF(
    #     xyz_channel=xyz_embedder.out_dim,
    #     viewdir_channel=viewdir_embedder.out_dim,
    #     hidden_dim=hidden_dim,
    # )

    return model, xyz_embedder, viewdir_embedder


def run_network(
    xyz_points: torch.Tensor,
    viewdir: torch.Tensor,
    model: Callable[[torch.Tensor], torch.Tensor],
    xyz_embedder: Embedder,
    viewdir_embedder: Embedder,
    chunk_size: int = 1024 * 64,
):
    """Embed inputs, batchify points to avoid OOM, and reshape output

    Parameters
    ----------
    xyz_points : torch.Tensor
        input points' 3D coordinate in World Coordinate system. [N_rays, N_samples,3]
    viewdir : torch.Tensor
        input data's viewing direction vector in World Coordiante system. Shape [N_rays,3]
    model : Callable[[torch.Tensor], torch.Tensor]
        Core NeRF model with MLP.
    xyz_embedder : Embedder
        Auxiliary positional embedding class for xyz coordinate.
    viewdir_embedder : Embedder
        Auxiliary positional embedding class for viewing direction vector.
    chunk_size : int, optional
        Chunk size to avoid Out of Memory, by default 1024*64

    Returns
    -------
    torch.Tensor
        Raw radiance data for each voxel. [N_rays, N_samples, 4] (rgbd data)
    """
    xyz_flat = xyz_points.flatten(start_dim=0, end_dim=-2)
    xyz_embedding = xyz_embedder.embed(xyz_flat)

    viewdir_embedding = viewdir_embedder.embed(viewdir)
    viewdir_embedding = viewdir_embedding[..., None, :].expand_as(xyz_flat)
    viewdir_flat_embedding = viewdir_embedding.flatten(start_dim=0, end_dim=-2)

    embedding = torch.cat([xyz_embedding, viewdir_flat_embedding], dim=-1)

    chunk_outputs: list[torch.Tensor] = []
    for i in range(0, embedding.size(0), chunk_size):
        chunk = embedding[i : i + chunk_size]
        chunk_outputs.append(model(chunk))
    output_flat = torch.cat(chunk_outputs, dim=0)
    output = output_flat.reshape(xyz_points.shape[:-1] + output_flat.shape[-1:])
    return output


def render_img(
    model: NeRF,
    width,
    height,
    focal_length,
    rays: tuple[torch.Tensor, ...],
    xyz_embedder: Embedder,
    viewdir_embedder: Embedder,
    chunk_size,
    near=0.0,
    far=1.0,
    N_coarse_sample: int = 64,
    N_fine_sample: int = 128,
    lin_disparity: bool = False,
    stratified: bool = True,
):
    """
    rays: [2,N_rays, 3]
    """
    rays_o, rays_d = rays
    # Simply speaking, viewdir is a normalized directional vector of rays.
    viewdir: torch.Tensor = rays_d / LA.vector_norm(
        rays_d,
        dim=-1,
        keepdim=True,
        dtype=torch.float32,
    )

    original_shape = rays_d.shape

    near = near * torch.ones_like(rays_d[..., :1])
    far = far * torch.ones_like(rays_d[..., :1])
    bundled_rays = torch.cat(
        [rays_o, rays_d, near, far],
        dim=-1,
    )  # [N_rays, 11]

    for i in range(0, bundled_rays.size(0), chunk_size):
        ray_batch = bundled_rays[i : i + chunk_size]
        viewdir_batch = viewdir[i : i + chunk_size]
        xyz_points, z_vals = sample_coarse_points(
            ray_batch,
            N_samples=N_coarse_sample,
            lin_disparity=lin_disparity,
            stratified=stratified,
        )
        raw = run_network(
            xyz_points,
            viewdir_batch,
            model,
            xyz_embedder=xyz_embedder,
            viewdir_embedder=viewdir_embedder,
        )
        result = integrate_ray(raw, z_vals, rays_d)

        # TODO: Run FINE network

    # merge results
