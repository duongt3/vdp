import os
import torch
from torch import nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from torchmetrics import Accuracy
from torchvision.datasets import ImageFolder

class ViT(pl.LightningModule):
    def __init__(self, ViT, num_classes=1000):
        super().__init__()
        self.model = ViT

        self.train_acc = Accuracy(task='multiclass', num_classes=num_classes)
        self.val_acc = Accuracy(task='multiclass', num_classes=num_classes)
        self.test_acc = Accuracy(task='multiclass', num_classes=num_classes)

        self.criterion = nn.CrossEntropyLoss()
    def training_step(self, batch, batch_idx):
        x, y = batch
        pred = self.model(x)
        loss = self.criterion(pred, y)
        self.train_acc(pred, y)
        self.log('train_acc', self.train_acc, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True, rank_zero_only=True)
        return loss

    @torch.enable_grad()
    def validation_step(self, batch, batch_idx):
        x, y = batch
        pred = self.model(x)
        loss = self.criterion(pred, y)
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True, rank_zero_only=True)
        self.val_acc(pred, y)
        self.log('val_acc', self.val_acc, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True, rank_zero_only=True)

    @torch.enable_grad()
    def test_step(self, batch, batch_idx):
        x, y = batch
        pred = self.model(x)
        loss = self.criterion(pred, y)
        self.log('test_loss', loss, rank_zero_only=True)
        self.test_acc(pred, y)
        self.log('test_acc', self.test_acc, on_step=True, on_epoch=True, sync_dist=True, rank_zero_only=True)
    
    @torch.enable_grad()
    def on_test_epoch_end(self):
        self.log('test_acc', self.test_acc, rank_zero_only=True)
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
    
    def test_dataloader(self):
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        test = ImageFolder('/data/ImageNet/val', transform=transform)
        return DataLoader(test, batch_size=128, num_workers=8, shuffle=False, pin_memory=True)
