"""This module includes the functions to generate the poses for novel view synthesis.
Codes in this module follos column-major matrix notation. (where )"""
import numpy as np
import torch


def change_axis() -> torch.Tensor:
    return torch.tensor(
        [
            [-1, 0, 0, 0],
            [0, 0, 1, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 1],
        ]
    ).type(torch.float32)


def translate(r: float) -> torch.Tensor:
    transformation_matrix = torch.tensor(
        [
            [1, 0, 0, 0],
            [0, 1, 0, r],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ]
    ).type(torch.float32)
    return transformation_matrix


def rotate_phi(phi: float) -> torch.Tensor:
    """Return transformation matrix along x axis.
    The unit of the rotation angle, phi, must be radian. (NOT degree)"""
    angle = angle = (phi - 90) / 180.0 * np.pi
    c, s = np.cos(angle), np.sin(angle)
    transformation_matrix = torch.tensor(
        [
            [1, 0, 0, 0],
            [0, c, -s, 0],
            [0, s, c, 0],
            [0, 0, 0, 1],
        ]
    ).type(torch.float32)
    return transformation_matrix


def rotate_theta(theta: float) -> torch.Tensor:
    """Return transformation matrix on the horizontal plane.
    The unit of the rotation angle, phi, must be radian. (NOT degree)"""
    angle = (90 - theta) / 180.0 * np.pi
    c, s = np.cos(angle), np.sin(angle)
    transformation_matrix = torch.tensor(
        [
            [c, -s, 0, 0],
            [s, c, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ]
    ).type(torch.float32)
    return transformation_matrix


def w2c(r, theta, phi):
    return change_axis @ translate(-r) @ rotate_phi(phi) @ rotate_theta(theta)


def c2w(r: float, theta: float, phi: float) -> torch.Tensor:
    """Return transformation matrix from camera coordinate system to world coordiante system.
    x-axis"""
    c2w = rotate_theta(theta).T @ rotate_phi(phi).T @ translate(r) @ change_axis()
    return c2w
