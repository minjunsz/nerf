"""This module includes the functions to generate the poses for novel view synthesis.
Codes in this module follos column-major matrix notation. (where )"""
import torch
import numpy as np


def translate_z(t: float) -> torch.Tensor:
    transformation_matrix = torch.Tensor(
        [[1, 0, 0, 0],
         [0, 1, 0, 0],
         [0, 0, 1, t],
         [0, 0, 0, 1]]
    ).type(torch.float32)
    return transformation_matrix


def rotate_phi(phi: float) -> torch.Tensor:
    """Return transformation matrix along x axis.
    The unit of the rotation angle, phi, must be radian. (NOT degree)"""
    transformation_matrix = torch.Tensor(
        [[1, 0, 0, 0],
         [0, np.cos(phi), np.sin(phi), 0],
         [0, -np.sin(phi), np.cos(phi), 0],
         [0, 0, 0, 1]]
    ).type(torch.float32)
    return transformation_matrix


def rotate_theta(theta: float) -> torch.Tensor:
    """Return transformation matrix on the horizontal plane.
    The unit of the rotation angle, phi, must be radian. (NOT degree)"""
    transformation_matrix = torch.Tensor(
        [[np.cos(theta), 0, np.sin(theta), 0],
         [0, 1, 0, 0],
         [-np.sin(theta), 0, np.cos(theta), 0],
         [0, 0, 0, 1]]
    ).type(torch.float32)
    return transformation_matrix


def calc_cam_to_world_transform(r: float, theta: float, phi: float) -> torch.Tensor:
    """Return transformation matrix from camera coordinate system to world coordiante system.
    x-axis """
    permute_y_z = torch.tensor([
        [1,0,0,0],
        [0,0,1,0],
        [0,1,0,0],
        [0,0,0,1]
    ]).type(torch.float32)
    c2w = permute_y_z @ rotate_theta(theta/180.*np.pi) @ rotate_phi(phi/180.*np.pi) @ translate_z(r)
    return c2w


if __name__ == "__main__":
    point = torch.tensor([2,1,-10,1]).type(torch.float32).unsqueeze(dim=1)
    transform = calc_cam_to_world_transform(4,0,90)