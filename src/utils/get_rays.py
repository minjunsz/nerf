"""This module provides the helper functions to generate rays casted through image pixels.
The code is written in column-major syntax(transformation matrix is multiplied at the left side of the vector.)
Pixel centering is NOT considered."""
import torch


def get_ray_directions(
    height: int, width: int, K: torch.Tensor, c2w: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate directional vectors of the casted rays through each pixels.
    Directional vectors are UNNORMALIZED vectors. This is just the vector from the camera origin(pinhole) to the pixel.
    Returned vectors live in WORLD COORDINATE.

    Parameters
    ----------
    height : int
        The number of pixels corresponds to the height of the image.
    width : int
        The number of pixels corresponds to the width of the image.
    K : torch.Tensor
        A transformation matrix from normalized image plane to pixel coordinate. This matrix consists of intrinsic parameters of the camera.
    c2w : torch.Tensor
        A transformation matrix from camera coordinate to world coordinate. This matrix adopts column-major manner.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        returned tuple is (rays_o, rays_d)
        rays_o : The origin of the rays in the WORLD COORDINATE. All rays share a common origin.
        rays_d : The direction of the rays in the WORLD COORDINATE.
    """
    x, y = torch.meshgrid(
        torch.linspace(0, width - 1, width),
        torch.linspace(0, height - 1, height),
        indexing="xy",
    )

    # normalized image plane.
    # In pixel coordinate, y+ is downward. In image coordinate, y+ is upward.
    u = (x - K[0][2]) / K[0][0]
    v = -(y - K[1][2]) / K[1][1]
    # As a convention, the normalized image plane is located in the z=-1 plane.
    # This is a directional vector lives in camera coordinate.
    directions = torch.stack([u, v, -torch.ones_like(u)], dim=-1)

    # Converted directional vectors into column vector before multiplying transformation matrix.
    # Convert it back to single-dimensional vector.
    rays_d = (c2w[:3, :3] @ directions[..., None]).unsqueeze(dim=-1)
    # Translate camera frame's origin to the world frame. It is the origin of all rays.
    rays_o = c2w[:3, -1].expand(rays_d.shape)
    return rays_o, rays_d
