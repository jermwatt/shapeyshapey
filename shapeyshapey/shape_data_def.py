from torch.utils.data import Dataset
import torch
import pandas as pd
import os
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
print(parent_dir)


class ShapeDataset(Dataset):
    def __init__(self,
                 data_path: str = parent_dir +
                 '/shape_dataset/test_shapes.csv',
                 transform=None):
        super().__init__()

        data = pd.read_csv(data_path)
        self.dataset = torch.tensor(data.to_numpy(), dtype=torch.float32)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> tuple:
        if self.transform:
            return self.transform(self.dataset[index])
        return self.dataset[index]
