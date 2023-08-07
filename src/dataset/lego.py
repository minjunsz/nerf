from typing import Literal
from dataclasses import dataclass
from pathlib import Path
import json

import imageio
import numpy as np
import torch
from torch.utils.data import Dataset

DATASPLIT = Literal['train', 'val', 'test']

@dataclass
class ViewData:
    imgs: torch.Tensor
    poses: torch.Tensor

class LegoDataset(Dataset):
    """This is a nerf dataset for synthetic lego model from blender."""
    
    width: int
    height: int
    focal_length: float
    
    def __init__(self, data_path: str, split: DATASPLIT="train", half_res: bool=False) -> None:
        super().__init__()
        self.data_path = Path(data_path)
        self.split = split
        self.half_res=half_res
        # Metadata includes camera parameters, image path and its camera pose.
        metadata_path = self.data_path / f"transforms_{split}.json"
        metadata = metadata_path.read_text()
        self.metadata = json.loads(metadata)
        
        sample_img = self.get_img_path(0)
        self.width, self.height = imageio.imread(sample_img).shape[0:2]
        self.focal_length = .5 * self.width / np.tan(.5 * self.metadata['camera_angle_x'])
        
    def get_img_path(self, index) -> Path:
        return self.data_path / f"{self.metadata['frames'][index]['file_path']}.png"
    
    def get_camera_params(self) -> tuple[int, int, float]:
        """return width & height of img, focal_length"""
        return self.width, self.height, self.focal_length
    
    def __len__(self) -> int:
        return len(self.metadata["frames"])
    
    def __getitem__(self, index) -> ViewData:
        img_path = self.get_img_path(index)
        img = torch.tensor(imageio.imread(img_path) / 255.).type(torch.float32)
        pose = torch.tensor(self.metadata['frames'][index]['transform_matrix']).type(torch.float32)
        return ViewData(img, pose)

    
if __name__ == "__main__":
    datapath = "/home/minjunsz/codes/nerf/data/nerf_synthetic/lego"
    dataset = LegoDataset(data_path=datapath, split="train")
    dataset[1]
    