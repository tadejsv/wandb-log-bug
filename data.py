import os

from torch.utils.data import DataLoader, random_split
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
import pytorch_lightning as pl

from hydra.utils import to_absolute_path


class MNISTDataModule(pl.LightningDataModule):
    def __init__(self, batch_size: int = 64, data_dir: str = "."):
        super().__init__()
        self.batch_size = batch_size
        self.data_dir = to_absolute_path(data_dir)
        self.cpu_cores = os.cpu_count()

    def setup(self, stage=None):
        self.mnist_test = MNIST(
            self.data_dir, train=False, download=True, transform=ToTensor()
        )
        mnist_full = MNIST(
            self.data_dir, train=True, download=True, transform=ToTensor()
        )
        self.mnist_train, self.mnist_val = random_split(mnist_full, [55000, 5000])

    def train_dataloader(self):
        return DataLoader(
            self.mnist_train, batch_size=self.batch_size, num_workers=self.cpu_cores
        )

    def val_dataloader(self):
        return DataLoader(
            self.mnist_val, batch_size=self.batch_size, num_workers=self.cpu_cores
        )

    def test_dataloader(self):
        return DataLoader(
            self.mnist_test, batch_size=self.batch_size, num_workers=self.cpu_cores
        )
