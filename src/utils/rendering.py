"""This module implements the volumetric rendering."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal

import torch
import torch.linalg as LA
import torch.nn.functional as F

from src.models.nerf import Embedder, NeRF


def sample_coarse_points(
    ray_batch: torch.Tensor, N_samples: int, lin_disparity: bool, stratified: bool
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample points for each rays. Returned points will be the inputs for the coarse NeRF network.

    Parameters
    ----------
    ray_batch : torch.Tensor
        Shape : [BATCH_SIZE, 8]. Information for rays.
        concatenation of rays_o(3) + rays_d(3) + near(1) + far(1)
    N_samples : int
        Number of samples per ray.
    lin_disparity : bool
        Determine how to sample the parameter of the ray function. (`t` value in `ray(t)=o+td`).
        If this is `False`, we linearly sample in depth (along z-axis).
        If this is `True`, we linearly sample in disparity rather than depth.
    stratified : bool
        If this is `True`, we perturb the regular intervals to implement stratified sampling.
        If this is `False`, we just use linearly sampled points as sample.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        Return (points, z_vals)
        points : Shape [N_rays, N_samples, 3]; xyz of sampled points in WORLD coordinate.
        z_vals : Shape [N_rays, N_samples]; parameters of sampled points for ray function.
    """

    N_rays = ray_batch.size(0)
    rays_o, rays_d = ray_batch[:, 0:3], ray_batch[:, 3:6]  # [N_rays, 3] each
    near, far = ray_batch[:, 6:7], ray_batch[:, 7:8]  # [N_rays, 1] each

    # evenly distributed samples
    t_vals = torch.linspace(0.0, 1.0, steps=N_samples)
    if lin_disparity:
        # sampling linearly in disparity rather than depth
        # disparity is inversely proportional to depth.
        z_vals = 1.0 / (1.0 / near * (1.0 - t_vals) + 1.0 / far * (t_vals))
    else:
        # distances from origin to evenly distributed points between near and far plane.
        z_vals = near * (1.0 - t_vals) + far * (t_vals)

    z_vals = z_vals.expand([N_rays, N_samples])

    # stratified sampling
    if stratified:
        # get intervals between samples
        mids = 0.5 * (z_vals[..., 1:] + z_vals[..., :-1])
        upper = torch.cat([mids, z_vals[..., -1:]], -1)
        lower = torch.cat([z_vals[..., :1], mids], -1)
        # stratified samples in those intervals
        t_rand = torch.rand(z_vals.shape)
        z_vals = lower + (upper - lower) * t_rand

    # Since rays_d is a vector up to image plane at z=-1,
    # we can simply use z-distance as a parameter of the rays; ray(t) = o+td.
    rays_o, rays_d = rays_o[..., None, :], rays_d[..., None, :]  # [N_rays, 1, 3]
    points = rays_o + rays_d * z_vals[..., :, None]  # [N_rays, N_samples, 3]

    return points, z_vals


def sample_fine_points(
    z_coarse_vals: torch.Tensor,
    weights: torch.Tensor,
    ray_batch: torch.Tensor,
    N_fine_samples: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Hierarchical sampling based of coarse sampling result. Use inverse transform sampling.

    Parameters
    ----------
    z_coarse_vals : torch.Tensor
        [N_rays, N_coarse_samples] coarse sampling results.
    weights : torch.Tensor
        [N_rays, N_coarse_samples] weights in volumetric rendering.
    ray_batch : torch.Tensor
        Shape : [BATCH_SIZE, 8]. Information for rays.
        concatenation of rays_o(3) + rays_d(3) + near(1) + far(1)
    N_fine_samples : int
        The number of fine samples.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        pts: [N_rays, N_coarse_samples + N_fine_samples, 3] sample points with hierarchical sampling.
        z_vals: [N_rays, N_coarse_samples + N_fine_samples] sampled ray parameters after hierarchical sampling.
    """
    rays_o, rays_d = ray_batch[:, 0:3], ray_batch[:, 3:6]  # [N_rays, 3] each

    bins = 0.5 * (
        z_coarse_vals[..., 1:] + z_coarse_vals[..., :-1]
    )  # [N_rays, N_coarse_samples-1]
    bins_weights = weights[..., 1:-1] + 1e-5  # [N_rays, N_coarse_samples-1]
    pdf = bins_weights / bins_weights.sum(
        dim=-1, keepdim=True
    )  # [N_rays, N_coarse_samples-1]
    cdf = pdf.cumsum(dim=-1)
    cdf = torch.cat(
        [torch.zeros_like(cdf[..., :1]), cdf],
        dim=-1,
    )  # [N_rays, N_coarse_samples]

    u = torch.rand((*cdf.shape[:-1], N_fine_samples))  # [N_rays, N_fine_samples]
    u = u.contiguous()
    # search the interval where `u` belongs.
    inds = torch.searchsorted(cdf, u, right=True)  # [N_rays, N_fine_samples]
    below = torch.max(torch.zeros_like(inds - 1), inds - 1)
    above = torch.min((cdf.shape[-1] - 1) * torch.ones_like(inds), inds)
    # fmt: off
    inds_g = torch.stack([below, above], -1)  # [N_rays, N_fine_samples, 2]
    matched_shape = [inds_g.shape[0], inds_g.shape[1], cdf.shape[-1]] # list([N_rays, N_fine_samples, N_coarse_samples])
    cdf_g = torch.gather(cdf.unsqueeze(1).expand(matched_shape), 2, inds_g) # [N_rays, N_fine_samples, 2]
    bins_g = torch.gather(bins.unsqueeze(1).expand(matched_shape), 2, inds_g) # [N_rays, N_fine_samples, 2]
    # fmt: on

    # interpolation within the sampled interval.
    denom = cdf_g[..., 1] - cdf_g[..., 0]
    denom = torch.where(denom < 1e-5, torch.ones_like(denom), denom)
    t = (u - cdf_g[..., 0]) / denom
    z_fine_samples = bins_g[..., 0] + t * (
        bins_g[..., 1] - bins_g[..., 0]
    )  # [N_rays, N_fine_samples]
    z_fine_samples = z_fine_samples.detach()

    sorted_z_vals = torch.cat([z_coarse_vals, z_fine_samples], dim=-1).sort(dim=-1)
    # [N_rays, N_coarse_samples + N_fine_samples]
    z_vals: torch.Tensor = sorted_z_vals[0]

    # [N_rays, N_coarse_samples + N_fine_samples, 3]
    pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., None]
    return pts, z_vals


@dataclass
class IntegrateResult:
    # fmt: off
    rgb_map: torch.Tensor           #  [num_rays, 3]. Estimated RGB color of a ray.
    disparity_map: torch.Tensor     #  [num_rays]. Disparity map. Inverse of depth map.
    acc_map: torch.Tensor           #  [num_rays]. Accumulated weights along each ray.
    depth_map: torch.Tensor         #  [num_rays]. Estimated distance to object.
    weights: torch.Tensor           #  [num_rays, num_samples]. Weights assigned to each sampled color.
    # fmt: on

    @staticmethod
    def merge_results(results: list[IntegrateResult]) -> IntegrateResult:
        return IntegrateResult(
            rgb_map=torch.cat([res.rgb_map for res in results], dim=0),
            disparity_map=torch.cat([res.disparity_map for res in results], dim=0),
            acc_map=torch.cat([res.acc_map for res in results], dim=0),
            depth_map=torch.cat([res.depth_map for res in results], dim=0),
            weights=torch.cat([res.weights for res in results], dim=0),
        )


def integrate_ray(
    raw: torch.Tensor,
    z_vals: torch.Tensor,
    rays_d: torch.Tensor,
    raw_noise_std: float = 0.0,
    white_bkgd: bool = False,
) -> IntegrateResult:
    """Transform NeRF models raw output to RGB pixel values by integrating rays; volumetric rendering.

    Parameters
    ----------
    raw : torch.Tensor
        [num_rays, num_samples along ray, 4]. Prediction from model.
    z_vals : torch.Tensor
        [num_rays, num_samples along ray]. Elapsed time t in the parametrized ray: ray(t)=o+td.
    rays_d : torch.Tensor
        [num_rays, 3]. Direction of each ray.
    raw_noise_std : float, optional
        standard deviation for alpha channel noise, by default 0.0
    white_bkgd : bool, optional
        Whether images have white background or not, by default False

    Returns
    -------
    IntegrateResult
        dataclass with (rgb_map, disparity_map, acc_map, weights, depth_map)
    """
    # This is Delta T in the parametrized ray: ray(t)=o+td.
    # append infinity distance for last element
    delta_t = z_vals[..., 1:] - z_vals[..., :-1]
    delta_t = torch.cat(
        [delta_t, torch.Tensor([1e10]).expand(delta_t[..., :1].shape)],
        dim=-1,
    )  # [N_rays, N_samples]

    # each ray has different speed because directional vector d is unnormalized.
    ray_speed = torch.norm(rays_d[..., None, :], dim=-1)

    dists = delta_t * ray_speed

    rgb_raw = torch.sigmoid(raw[..., :3])  # [N_rays, N_samples, 3]
    alpha_raw = raw[..., 3]  # [N_rays, N_samples]
    # perturb alpha channel with Gaussian noise
    if raw_noise_std > 0.0:
        noise = torch.randn_like(alpha_raw) * raw_noise_std
        alpha_raw = alpha_raw + noise
    alpha = 1.0 - torch.exp(-F.relu(alpha_raw) * dists)  # [N_rays, N_samples]

    # Math: weights_i = alpha_i * prod^{i-1}_{j=1} (1-alpha_j)
    accumulated_transmittance = (1.0 - alpha + 1e-10).cumprod(dim=-1)
    accumulated_transmittance = torch.cat(
        [
            torch.ones_like(accumulated_transmittance[..., :1]),
            accumulated_transmittance[..., :-1],
        ],
        dim=-1,
    )
    # weights = alpha * tf.math.cumprod(1.-alpha + 1e-10, -1, exclusive=True)
    weights = alpha * accumulated_transmittance  # [N_rays, N_samples]

    rgb_map = torch.sum(weights[..., None] * rgb_raw, dim=-2)  # [N_rays, 3]
    depth_map = torch.sum(weights * z_vals, -1)
    disparity_map = 1.0 / torch.max(
        1e-10 * torch.ones_like(depth_map), depth_map / torch.sum(weights, -1)
    )
    acc_map = torch.sum(weights, -1)

    # If ray didn't hit any object (low accumulated weights),
    # make the pixel white by adding same value to all RGB channels.
    if white_bkgd:
        rgb_map = rgb_map + (1.0 - acc_map[..., None])

    return IntegrateResult(
        rgb_map=rgb_map,
        disparity_map=disparity_map,
        acc_map=acc_map,
        depth_map=depth_map,
        weights=weights,
    )


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
    xyz_embedding = xyz_embedder.embed(xyz_flat)  # [N_rays*N_samples,3]

    viewdir_expand = viewdir[..., None, :].expand_as(xyz_points)
    viewdir_flat = viewdir_expand.flatten(start_dim=0, end_dim=-2)
    viewdir_embedding = viewdir_embedder.embed(viewdir_flat)

    embedding = torch.cat([xyz_embedding, viewdir_embedding], dim=-1)

    chunk_outputs: list[torch.Tensor] = []
    for i in range(0, embedding.size(0), chunk_size):
        chunk = embedding[i : i + chunk_size]
        chunk_outputs.append(model(chunk))
    output_flat = torch.cat(chunk_outputs, dim=0)
    output = output_flat.reshape(xyz_points.shape[:-1] + output_flat.shape[-1:])
    return output


def render_img(
    coarse_model: NeRF,
    fine_model: NeRF,
    rays_o: torch.Tensor,
    rays_d: torch.Tensor,
    xyz_embedder: Embedder,
    viewdir_embedder: Embedder,
    chunk_size: int = 1024 * 32,
    near=0.0,
    far=1.0,
    N_coarse_sample: int = 64,
    N_fine_sample: int = 128,
    lin_disparity: bool = False,
    stratified: bool = True,
) -> dict[Literal["coarse", "fine"], IntegrateResult]:
    """Render an 2D image with the given model.

    Parameters
    ----------
    coarse_model : NeRF
        Core NeRF model for coarse-grained samples
    fine_model : NeRF
        Core NeRF model for fine-grained samples (hierarchical sampling)
    rays_o : torch.Tensor
        [N_rays, 3] origin of rays in world coordinate
    rays_d : torch.Tensor
        [N_rays, 3] unnormalized directional vector of rays in world coordinate.
    xyz_embedder : Embedder
        Auxiliary embedder object for positional encoding of XYZ position in world coordinate.
    viewdir_embedder : Embedder
        Auxiliary embedder object for positional encoding of viewing direction vector in world coordinate.
    chunk_size : int, optional
        Size of batch to avoid OOM, by default 1024*32
    near : float, optional
        Near cutoff of the viewing frustum, by default 0.0
    far : float, optional
        Far cutoff of the viewing frustum, by default 1.0
    N_coarse_sample : int, optional
        The number of samples for coarse sampling, by default 64
    N_fine_sample : int, optional
        The number of samples for hierarchical sampling, by default 128
    lin_disparity : bool, optional
        How to sample coarse samples. If true, points are linearly sampled in disparity rather than depth, by default False
    stratified : bool, optional
        Whether to apply stratified sampling for coarse samples or not. If false, points are sampled deterministically, by default True

    Returns
    -------
    dict[Literal["coarse", "fine"], IntegrateResult]
        Results of volumetric integrations.
    """
    # Simply speaking, viewdir is a normalized directional vector of rays.
    viewdir: torch.Tensor = rays_d / LA.vector_norm(
        rays_d,
        dim=-1,
        keepdim=True,
        dtype=torch.float32,
    )

    near = near * torch.ones_like(rays_d[..., :1])
    far = far * torch.ones_like(rays_d[..., :1])
    bundled_rays = torch.cat(
        [rays_o, rays_d, near, far],
        dim=-1,
    )  # [N_rays, 11]

    coarse_results: list[IntegrateResult] = []
    fine_results: list[IntegrateResult] = []

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
            coarse_model,
            xyz_embedder=xyz_embedder,
            viewdir_embedder=viewdir_embedder,
        )
        coarse_result = integrate_ray(raw, z_vals, rays_d)

        # TODO: Run FINE network
        xyz_points_fine, z_vals_fine = sample_fine_points(
            z_vals,
            coarse_result.weights,
            ray_batch,
            N_fine_sample,
        )
        raw_fine = run_network(
            xyz_points_fine,
            viewdir_batch,
            fine_model,
            xyz_embedder,
            viewdir_embedder,
        )
        fine_result = integrate_ray(raw_fine, z_vals_fine, rays_d)

        coarse_results.append(coarse_result)
        fine_results.append(fine_result)

    return {
        "coarse": IntegrateResult.merge_results(coarse_results),
        "fine": IntegrateResult.merge_results(fine_results),
    }
