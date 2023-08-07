"""This module provides the helper functions to generate rays casted through image pixels.
The code is written in column-major syntax(transformation matrix is multiplied at the left side of the vector.)
Pixel centering is NOT considered."""
import torch


def get_ray_directions(
    height: int, width: int, focal_length: float, K: torch.Tensor
) -> torch.Tensor:
    """Generate directional vectors of the casted rays through each pixels.
    Directional vectors are UNNORMALIZED vectors. This is just the vector from the origin of the camera coordinate(pinhole) to the pixel.

    Parameters
    ----------
    height : int
        The number of pixels corresponds to the height of the image.
    width : int
        The number of pixels corresponds to the width of the image.
    focal_length : float
        focal_length of the camera.
    K : torch.Tensor
        A transformation matrix from normalized image plane to pixel coordinate. This matrix consists of intrinsic parameters of the camera.

    Returns
    -------
    torch.Tensor
        Shape: (H,W,3), the direction of the rays in the CAMERA COORDINATE.
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
    directions = torch.stack([u, v, -torch.ones_like(u)], dim=-1)
    return directions
