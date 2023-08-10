"""This module implements the volumetric rendering."""
from dataclasses import dataclass

import torch
import torch.nn.functional as F


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


# fmt: off
@dataclass
class IntegrateResult:
    rgb_map: torch.Tensor           #  [num_rays, 3]. Estimated RGB color of a ray.
    disparity_map: torch.Tensor     #  [num_rays]. Disparity map. Inverse of depth map.
    acc_map: torch.Tensor           #  [num_rays]. Accumulated weights along each ray.
    depth_map: torch.Tensor         #  [num_rays]. Estimated distance to object.
    weights: torch.Tensor           #  [num_rays, num_samples]. Weights assigned to each sampled color.
# fmt: on


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
