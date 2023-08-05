from typing import Literal
from torch.utils.data import Dataset

DATASPLIT = Literal['train', 'val', 'test']

class LegoData(Dataset):
    """This is a nerf dataset for synthetic lego model from blender."""
    
    def __init__(self, data_path: str, split: DATASPLIT="train") -> None:
        super().__init__()
        