from torch.utils.data import Dataset
import torch
import os
import sys
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)
library_path = parent_dir + '/shape_generator'
sys.path.append(library_path)
from shape_generator.simple_square_tiling import read_all_image_pairs_from_idx


class ShapeDataset(Dataset):
    def __init__(self,
                 data_path: str,
                 transform=None):
        super().__init__()
        # Example usage
        data = read_all_image_pairs_from_idx(data_path)
        self.dataset = [[torch.tensor(v[0], dtype=torch.float32),
                         torch.tensor(v[1], dtype=torch.float32)] 
                        for v in data]
        self.transform = transform

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> tuple:
        if self.transform:
            return self.transform(self.dataset[index][0]), \
                   self.transform(self.dataset[index][1])

        return self.dataset[index][0], self.dataset[index][1]
