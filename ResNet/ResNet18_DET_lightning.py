import os

import torchvision.transforms

import torch.nn as nn
import utils.vdp as vdp
import torch
import wandb
import argparse
import pytorch_lightning as pl
from torch.nn.utils import prune
from torchmetrics import Accuracy
import torchvision.models as models
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from torchvision.transforms import transforms
from torchvision.transforms import RandAugment, AutoAugment
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from torch.optim import lr_scheduler
import torch.nn.functional as F

def create_model():
    model = torchvision.models.resnet18(pretrained=False, num_classes=10)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    model.maxpool = nn.Identity()
    return model

class det_cifar10(pl.LightningModule):
    def __init__(self, lr=0.05, wd=0.0005, pt=10, optim='adam', sched='onecycle', batch_size=128):
        super().__init__()
        self.lr = lr
        self.wd = wd
        self.pt = pt
        self.optim = optim
        self.sched = sched
        self.batch_size = batch_size
        self.train_acc = Accuracy()
        self.val_acc = Accuracy()
        self.test_acc = Accuracy()
        self.criterion = torch.nn.CrossEntropyLoss()
        # backbone = models.resnet18(pretrained=False)
        # num_filters = backbone.fc.in_features
        # layers = list(backbone.children())[:-1]
        # self.feature_extractor = torch.nn.Sequential(*layers)
        self.model = create_model()
        num_target_classes = 10
        # self.classifier = torch.nn.Linear(num_filters, num_target_classes)

    def train_dataloader(self):
        transform = transforms.Compose([
            # AutoAugment(policy=torchvision.transforms.AutoAugmentPolicy.CIFAR10),
            torchvision.transforms.RandomCrop(32, padding=4),
            torchvision.transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        train = CIFAR10(os.getcwd(), train=True, download=True, transform=transform)
        return DataLoader(train, batch_size=self.batch_size, num_workers=8, shuffle=True, pin_memory=True)

    def val_dataloader(self):
        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
        test = CIFAR10(os.getcwd(), train=False, download=True, transform=transform)
        return DataLoader(test, batch_size=self.batch_size, num_workers=8, shuffle=False, pin_memory=True)

    def test_dataloader(self):
        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
        test = CIFAR10(os.getcwd(), train=False, download=True, transform=transform)
        return DataLoader(test, batch_size=self.batch_size, num_workers=8, shuffle=False, pin_memory=True)

    def configure_optimizers(self):
        if self.optim == "adam":
            optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.wd, amsgrad=True)
        elif self.optim == "sgd":
            optimizer = torch.optim.SGD(self.parameters(), lr=self.lr,
                        momentum=0.9, weight_decay=self.wd)
        else:
            raise ValueError('Invalid optimizer: ' + self.optim)

        if self.sched == "multisteplr":
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [40, 80, 120], 0.05, verbose=True)
        elif self.sched == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
        elif self.sched == "plateau":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=self.pt, verbose=True)
        elif self.sched == "onecycle":
            steps_per_epoch = 45000 // self.batch_size
            scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, 0.1, epochs=self.trainer.max_epochs, steps_per_epoch=steps_per_epoch)
        else:
            raise ValueError('Invalid scheduler: ' + self.sched)

        return {
            'optimizer': optimizer,
            'lr_scheduler': scheduler,
            'monitor': 'loss'
        }

    def get_progress_bar_dict(self):
        items = super().get_progress_bar_dict()
        items.pop("v_num", None)
        return items

    def forward(self, x):
        # representations = self.feature_extractor(x).flatten(1)
        # x = self.classifier(representations)
        # return x
        out = self.model(x)
        return F.log_softmax(out, dim=1)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        self.train_acc(logits.softmax(dim=-1), y)
        self.log('loss', loss)
        self.log('train_acc', self.train_acc, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        self.log('val_loss', loss)
        self.val_acc(logits.softmax(dim=-1), y)
        self.log('val_acc', self.val_acc, on_step=True, on_epoch=True, prog_bar=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        self.log('test_loss', loss)
        self.test_acc(logits.softmax(dim=-1), y)
        self.log('test_acc', self.test_acc, on_step=True, on_epoch=True)

    def on_test_epoch_end(self):
        self.log('test_acc', self.test_acc)
