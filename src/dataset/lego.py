import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import imageio.v2 as imageio
import numpy as np
import torch
import torchvision.transforms.functional as F
from torch.utils.data import Dataset

from src.utils.get_rays import get_rays

DATASPLIT = Literal["train", "val", "test"]


@dataclass
class ViewData:
    imgs: torch.Tensor
    poses: torch.Tensor
    width: int
    height: int
    K: torch.Tensor


class LegoDataset(Dataset):
    """This is a nerf dataset for synthetic lego model from blender."""

    current_image: np.ndarray

    def __init__(
        self,
        data_path: Path,
        split: DATASPLIT = "train",
        half_res: bool = False,
        white_background: bool = True,
    ) -> None:
        super().__init__()
        self.data_path = data_path
        self.split = split
        self.half_res = half_res
        self.white_background = white_background
        self.frames: list[Any]
        metadata_path = self.data_path / f"transforms_{split}.json"
        with metadata_path.open() as f:
            metadata = json.load(f)
            sample_img = Path(data_path) / f"{metadata['frames'][0]['file_path']}.png"
            self.height, self.width = imageio.imread(sample_img).shape[0:2]
            focal_length = 0.5 * self.width / np.tan(0.5 * metadata["camera_angle_x"])
            if half_res is True:
                self.height, self.width, focal_length = (
                    self.height // 2,
                    self.width // 2,
                    focal_length / 2.0,
                )
            self.K = torch.tensor(
                [
                    [focal_length, 0, 0.5 * self.width],
                    [0, focal_length, 0.5 * self.height],
                    [0, 0, 1],
                ]
            )
            self.frames = metadata["frames"]

    def __len__(self) -> int:
        return len(self.frames)

    def __getitem__(self, index) -> ViewData:
        img_path = self.data_path / f"{self.frames[index]['file_path']}.png"
        img = torch.tensor(imageio.imread(img_path) / 255.0).type(torch.float32)
        pose = torch.tensor(self.frames[index]["transform_matrix"]).type(torch.float32)

        if self.half_res:
            H, W = img.shape[0:2]
            permuted = img.permute(
                2, 0, 1
            )  # resize in torchvision expects [...,H,W] shape
            img = F.resize(permuted, [H // 2, W // 2], antialias=True).permute(1, 2, 0)

        if self.white_background:
            img = img[..., :3] * img[..., -1:] + (1.0 - img[..., -1:])
        else:
            img = img[..., :3]

        return ViewData(img, pose, self.width, self.height, self.K)

    def get_iterable_rays(
        self,
        pre_crop: bool = False,
        pre_crop_frac: float = 0.5,
        batch_size: int = 2048,
        randomize: bool = True,
    ):
        sample_idx = np.random.randint(len(self))
        data = self[sample_idx]
        img, pose = data.imgs, data.poses
        self.current_image = img.cpu().numpy()
        rays_o, rays_d = get_rays(self.height, self.width, self.K, c2w=pose)

        if pre_crop:
            dH = int(self.height // 2 * pre_crop_frac)
            dW = int(self.width // 2 * pre_crop_frac)
            coords = torch.stack(
                torch.meshgrid(
                    torch.linspace(
                        self.height // 2 - dH, self.height // 2 + dH - 1, 2 * dH
                    ),
                    torch.linspace(
                        self.width // 2 - dW, self.width // 2 + dW - 1, 2 * dW
                    ),
                    indexing="ij",
                ),
                dim=-1,
            )
        else:
            coords = torch.stack(
                torch.meshgrid(
                    torch.linspace(0, self.height - 1, self.height),
                    torch.linspace(0, self.width - 1, self.width),
                    indexing="ij",
                ),
                dim=-1,
            )  # (H, W, 2)

        coords = coords.reshape(-1, 2)  # (H * W, 2)
        if randomize:
            indices = np.random.permutation(coords.size(0))
        else:
            indices = np.arange(coords.size(0))

        for i in range(0, coords.size(0), batch_size):
            batch_idx = indices[i : i + batch_size]
            coords_batch = coords[batch_idx].long()
            rays_o_batch = rays_o[
                coords_batch[:, 0], coords_batch[:, 1]
            ]  # (batch_size, 3)
            rays_d_batch = rays_d[
                coords_batch[:, 0], coords_batch[:, 1]
            ]  # (batch_size, 3)
            gt_pixels = img[coords_batch[:, 0], coords_batch[:, 1]]  # (batch_size, 3)

            yield (rays_o_batch, rays_d_batch, gt_pixels)
