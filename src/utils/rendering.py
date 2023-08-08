"""This module implements the volumetric rendering."""
import torch


def sample_coarse_points(
    ray_batch: torch.Tensor, N_samples: int, lin_disparity: bool, stratified: bool
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample points for each rays. Returned points will be the inputs for the coarse NeRF network.

    Parameters
    ----------
    ray_batch : torch.Tensor
        Shape : [BATCH_SIZE, 11]. Information for rays.
        concatenation of rays_o(3) + rays_d(3) + near(1) + far(1) + viewdir(3)
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
    if stratified > 0.0:
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
