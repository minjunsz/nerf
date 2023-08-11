from typing import Literal, Any
from dataclasses import dataclass
from pathlib import Path
import json

import imageio.v2 as imageio
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as F

DATASPLIT = Literal["train", "val", "test"]


@dataclass
class ViewData:
    imgs: torch.Tensor
    poses: torch.Tensor


def read_camera_params(
    data_path: str, split: DATASPLIT = "train", half_res: bool = False
) -> tuple[int, int, float]:
    """Read camera parameters from the metadata file.
    Returns (height, width, focal_length) tuple."""
    metadata_path = Path(data_path) / f"transforms_{split}.json"
    metadata = metadata_path.read_text()
    metadata = json.loads(metadata)
    sample_img = Path(data_path) / f"{metadata['frames'][0]['file_path']}.png"
    height, width = imageio.imread(sample_img).shape[0:2]
    focal_length = 0.5 * width / np.tan(0.5 * metadata["camera_angle_x"])
    if half_res is True:
        height, width, focal_length = height // 2, width // 2, focal_length / 2.0
    return height, width, focal_length


class LegoDataset(Dataset):
    """This is a nerf dataset for synthetic lego model from blender."""

    def __init__(
        self, data_path: Path, split: DATASPLIT = "train", half_res: bool = False
    ) -> None:
        super().__init__()
        self.data_path = data_path
        self.split = split
        self.half_res = half_res
        self.frames: list[Any]
        metadata_path = self.data_path / f"transforms_{split}.json"
        with metadata_path.open() as f:
            metadata = json.load(f)
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

        return ViewData(img, pose)


if __name__ == "__main__":
    datapath = Path("/home/minjun/codes/nerf/data/nerf_synthetic/lego")
    dataset = LegoDataset(data_path=datapath, split="train", half_res=True)
