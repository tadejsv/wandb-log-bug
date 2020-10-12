import os

from torch.utils.data import DataLoader, random_split
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor

from hydra.utils import to_absolute_path


def train_dataloader():

    BATCH_SIZE = 16
    dataset = MNIST(
        to_absolute_path("."), train=True, download=True, transform=ToTensor()
    )

    return DataLoader(dataset, batch_size=BATCH_SIZE, num_workers=os.cpu_count())
