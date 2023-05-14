from pytorch_lightning.utilities.types \
    import EVAL_DATALOADERS, TRAIN_DATALOADERS
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.data import random_split
import pytorch_lightning as pl

# parent of current file directory
import os
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class MnistDataModule(pl.LightningDataModule): 
    def __init__(self,
                 data_dir: str = parent_dir + '/dataset',
                 batch_size: int = 64,
                 num_workers: int = 4):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers

    def prepare_data(self) -> None:
        datasets.MNIST(self.data_dir, train=True, download=True)
        datasets.MNIST(self.data_dir, train=False, download=True)

    def setup(self, stage=None):
        # define entire dataset ,transform
        entire_dataset = datasets.MNIST(
            root=self.data_dir, train=True,
            transform=transforms.Compose([
                # transforms.RandomVerticalFlip(),
                # transforms.RandomHorizontalFlip(),
                transforms.ToTensor()
            ]),
            download=False
        )

        # define test dataset
        self.test_ds = datasets.MNIST(
            root=self.data_dir, 
            train=False,
            transform=transforms.ToTensor(),
            download=False
        )

        # split dataset
        self.train_ds, self.val_ds = random_split(entire_dataset, [50000, 10000])

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(self.train_ds, 
                          batch_size=self.batch_size, 
                          num_workers=self.num_workers,
                          shuffle=True)

    def val_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(self.val_ds,
                          batch_size=self.batch_size,
                          num_workers=self.num_workers,
                          shuffle=False)

    def test_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(self.test_ds,
                          batch_size=self.batch_size,
                          num_workers=self.num_workers,
                          shuffle=False)
