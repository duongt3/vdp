from typing import Optional
import medmnist
from medmnist import INFO, Evaluator
import torch.utils.data as data
import torchvision.transforms as transforms
from pytorch_lightning import LightningModule, Trainer, seed_everything
import pytorch_lightning as pl
from torch.utils.data import random_split, DataLoader
import numpy as np

# Note - you must have torchvision installed for this example
from torchvision.datasets import MNIST
from torchvision import transforms

seed_everything(7)

class MedMnistDataModule(pl.LightningDataModule):
    def __init__(self, batch_size: int = 32):
        super().__init__()
        self.batch_size = batch_size

    def setup(self, stage: Optional[str] = None):
        # data_flag = 'pathmnist'
        # data_flag = 'breastmnist'
        data_flag = 'retinamnist'
        download = True

        self.BATCH_SIZE = 128

        info = INFO[data_flag]

        DataClass = getattr(medmnist, info['python_class'])

        # preprocessing
        data_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[.5], std=[.5])
        ])

        # load the data
        self.train_dataset = DataClass(split='train', transform=data_transform, download=download)
        self.test_dataset = DataClass(split='test', transform=data_transform, download=download)

        self.pil_dataset = DataClass(split='train', download=download)

        # Convert labels to ordinal
        # self.train_dataset.labels = np.array(list(map(convert_to_ordinal, self.train_dataset.labels)))
        # self.test_dataset.labels = np.array(list(map(convert_to_ordinal, self.test_dataset.labels)))

    def train_dataloader(self):
        return data.DataLoader(dataset=self.train_dataset, batch_size=self.BATCH_SIZE, shuffle=True)

    def val_dataloader(self):
        return data.DataLoader(dataset=self.train_dataset, batch_size=2*self.BATCH_SIZE, shuffle=False)

    def test_dataloader(self):
        return data.DataLoader(dataset=self.test_dataset, batch_size=2*self.BATCH_SIZE, shuffle=False)

    def predict_dataloader(self):
        return data.DataLoader(dataset=self.test_dataset, batch_size=2*self.BATCH_SIZE, shuffle=False)
