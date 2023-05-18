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
from pytorch_lightning import LightningModule, Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from MedMnistDataModule import MedMnistDataModule
from medmnist.evaluator import getAUC, getACC

BATCH_SIZE = 128

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--lr', help='Learning Rate', type=float, default=5e-2, required=False)
    parser.add_argument(
        '--weight_decay', help='Weight Decay', type=float, default=0.0005, required=False)
    parser.add_argument(
        '--no_epochs', help='Number of Epochs', type=int, default=100, required=False)
    parser.add_argument(
        '--optim', help='Optimizer Type (adam, sgd)', type=str, default="sgd")
    parser.add_argument(
        '--sched', help='Scheduler Type (multisteplr, cosine, plateau, onecycle)', type=str, default="cosine")
    parser.add_argument(
        '--patience', help='Number of Epochs until sched decreases', type=int, default=5, required=False)
    parser.add_argument(
        '--batch_size', help='Batch Size', type=int, default=128)
    parser.add_argument(
        '--num_runs', help="Number of models to train", type=int, default=1)
    args = parser.parse_args()
    return args

def create_model():
    model = torchvision.models.resnet18(pretrained=False, num_classes=5)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    model.maxpool = nn.Identity()
    return model

class LitResnet(LightningModule):
    def __init__(self, wd=0.0005, pt=10, optim='adam', sched='onecycle', lr=0.05):
        super().__init__()
        self.lr = lr
        self.wd = wd
        self.pt = pt
        self.optim = optim
        self.sched = sched
        self.save_hyperparameters()
        self.train_acc = Accuracy()
        self.val_acc = Accuracy()
        self.test_acc = Accuracy()
        self.model = create_model()
        self.criterion = torch.nn.CrossEntropyLoss()

    @torch.enable_grad()
    def training_step(self, batch, batch_idx):
        x, y = batch
        y = torch.squeeze(y)
        logits = self(x)
        loss = self.criterion(logits, y)
        # self.train_acc(logits.softmax(dim=-1), y)
        self.log('loss', loss)
        acc = getACC(y.detach().cpu().numpy(), logits.sigmoid().detach().cpu().numpy(), "lol")
        auc = getAUC(y.detach().cpu().numpy(), logits.sigmoid().detach().cpu().numpy(), "lol" )
        self.log('train_acc', acc, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_auc', auc, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    @torch.enable_grad()
    def forward(self, x):
        out = self.model(x)
        # return F.log_softmax(out, dim=1)
        return out

    @torch.enable_grad()
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y = torch.squeeze(y)
        logits = self(x)
        loss = self.criterion(logits, y)
        self.log('val_loss', loss)
        # self.val_acc(logits.sigmoid(), y)
        acc = getACC(y.detach().cpu().numpy(), logits.sigmoid().detach().cpu().numpy(), "lol")
        auc = getAUC(y.detach().cpu().numpy(), logits.sigmoid().detach().cpu().numpy(), "lol" )
        self.log('val_acc', acc, on_step=True, on_epoch=True, prog_bar=True)
        self.log('val_auc', auc, on_step=True, on_epoch=True, prog_bar=True)

    @torch.enable_grad()
    def test_step(self, batch, batch_idx):
        x, y = batch
        y = torch.squeeze(y)
        logits = self(x)
        loss = self.criterion(logits, y)
        self.log('test_loss', loss)
        # self.test_acc(logits.sigmoid(), y)
        acc = getACC(y.detach().cpu().numpy(), logits.sigmoid().detach().cpu().numpy(), "lol")
        auc = getAUC(y.detach().cpu().numpy(), logits.sigmoid().detach().cpu().numpy(), "lol" )
        self.log('test_acc', acc, on_step=True, on_epoch=True)
        self.log('test_auc', auc, on_step=True, on_epoch=True)

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
        else:
            raise ValueError('Invalid scheduler: ' + self.sched)

        return {
            'optimizer': optimizer,
            'lr_scheduler': scheduler,
            'monitor': 'loss'
        }

def main():
    cmd = parse_args()
    wandb.init(config=cmd, project="med_det_resnet18_sweep")
    config = wandb.config
    model = LitResnet(wd=config.weight_decay, pt=config.patience, optim=config.optim, sched=config.sched, lr=config.lr)
   
    wandb_logger = WandbLogger(log_model="all")
    checkpoint_callback = ModelCheckpoint(monitor="val_acc", mode="max")
    early_stop_callback = EarlyStopping(monitor="loss", min_delta=0.05, patience=3, verbose=True, mode="min")
    trainer = Trainer(
        gpus=[6],
        max_epochs=300,
        accelerator=None,
        check_val_every_n_epoch=5,
        # devices=1 if torch.cuda.is_available() else None,  # limiting got iPython runs
        logger=wandb_logger,
        callbacks=[LearningRateMonitor(logging_interval="step"), checkpoint_callback, early_stop_callback],
    )

    medMnist = MedMnistDataModule(batch_size=config.batch_size)
    trainer.fit(model, medMnist)
    trainer.test(model, datamodule=medMnist)

if __name__ =='__main__':
    main()