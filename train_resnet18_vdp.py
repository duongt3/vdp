import os
import argparse
import math
import wandb
import pandas as pd
import seaborn as sn
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import torchvision
import glob
from pl_bolts.datamodules import CIFAR10DataModule
from pl_bolts.transforms.dataset_normalizations import cifar10_normalization
from pytorch_lightning import LightningModule, Trainer, seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.loggers import CSVLogger
from torch.optim.lr_scheduler import OneCycleLR
from torch.optim.swa_utils import AveragedModel, update_bn
from torchmetrics.functional import accuracy
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
import ResNet_VDP.ResNet_vdp as ResNet_vdp
import utils.vdp as vdp
from torchmetrics import Accuracy

seed_everything(7)

PATH_DATASETS = os.environ.get("PATH_DATASETS", ".")
BATCH_SIZE = 256 if torch.cuda.is_available() else 64
NUM_WORKERS = int(os.cpu_count() / 2)

train_transforms = torchvision.transforms.Compose(
    [
        torchvision.transforms.RandomCrop(32, padding=4),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.ToTensor(),
        cifar10_normalization(),
    ]
)

test_transforms = torchvision.transforms.Compose(
    [
        torchvision.transforms.ToTensor(),
        cifar10_normalization(),
    ]
)

cifar10_dm = CIFAR10DataModule(
    data_dir=PATH_DATASETS,
    batch_size=BATCH_SIZE,
    num_workers=NUM_WORKERS,
    train_transforms=train_transforms,
    test_transforms=test_transforms,
    val_transforms=test_transforms,
)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--lr', help='Learning Rate', type=float, default=5e-2, required=False)
    parser.add_argument(
        '--wd', help='Weight Decay', type=float, default=0.0005, required=False)
    # parser.add_argument(
    #     '--no_epochs', help='Number of Epochs', type=int, default=500, required=False)
    parser.add_argument(
        '--optim', help='Optimizer Type (adam, sgd)', type=str, default="adam")
    parser.add_argument(
        '--sched', help='Scheduler Type (multisteplr, cosine, plateau, onecycle)', type=str, default="onecycle")
    parser.add_argument(
        '--patience', help='Number of Epochs until sched decreases', type=int, default=10, required=False)
    # parser.add_argument(
    #     '--batch_size', help='Batch Size', type=int, default=128)
    # parser.add_argument(
    #     '--num_runs', help="Number of models to train", type=int, default=1)
    parser.add_argument(
        '--psi', help="Hyperparameter for scaling nll loss term", type=int, default=700)
    parser.add_argument(
        '--tau', help="Hyperparameter for tuning KL loss term", type=float, default=0.01)
    args = parser.parse_args()
    return args


def create_model():
    model = ResNet_vdp.resnet18(pretrained=False, num_classes=10)
    model.conv1 = vdp.Conv2d(3, 64, kernel_size=3, stride=(1, 1), padding=(1, 1), bias=False, input_flag=True)
    model.maxpool = vdp.Identity()
    return model

class LitResnet(LightningModule):
    def __init__(self, psi=700, tau=0.01, wd=0.0005, pt=10, optim='adam', sched='onecycle', lr=0.05):
        super().__init__()
        self.lr = lr
        self.wd = wd
        self.pt = pt
        self.optim = optim
        self.sched = sched
        self.psi = psi
        self.tau = tau
        self.save_hyperparameters()
        self.train_acc = Accuracy()
        self.val_acc = Accuracy()
        self.test_acc = Accuracy()
        self.model = create_model()

    @torch.enable_grad()
    def training_step(self, batch, batch_idx):
        x, y = batch
        mu, sigma = self(x)
        loss = self.get_loss(mu, sigma, y)
        self.log("loss", loss)
        self.train_acc(mu.softmax(dim=-1), y)
        self.log('train_acc', self.train_acc, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    @torch.enable_grad()
    def forward(self, x):
        mu, sigma = self.model(x)
        return mu, sigma

    def get_loss(self, mu, sigma, y):
        log_det, nll = vdp.ELBOLoss(mu, sigma, y)
        kl = vdp.gather_kl(self.model)
        # if self.alpha is None:
        #     self.alpha, self.tau = vdp.scale_hyperp(log_det, nll, kl)
        # loss = self.alpha * log_det + nll + self.tau * sum(kl)
        loss = log_det + self.psi*nll + self.tau*sum(kl)
        self.log("log_det", log_det)
        self.log("nll", self.psi*nll)
        self.log("tau", self.tau*sum(kl))
        return loss

    def evaluate(self, batch, stage=None):
        x, y = batch
        mu, sigma = self(x)
        loss = self.get_loss(mu, sigma, y)
        acc = accuracy(mu.softmax(dim=-1), y)

        if stage:
            self.log(f"{stage}_loss", loss, prog_bar=True)
            self.log(f"{stage}_acc", acc, prog_bar=True)

    @torch.enable_grad()
    def validation_step(self, batch, batch_idx):
        self.evaluate(batch, "val")

    @torch.enable_grad()
    def test_step(self, batch, batch_idx):
        self.evaluate(batch, "test")

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
            steps_per_epoch = 45000 // BATCH_SIZE
            scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, 0.1, epochs=self.trainer.max_epochs, steps_per_epoch=steps_per_epoch)
        else:
            raise ValueError('Invalid scheduler: ' + self.sched)

        return {
            'optimizer': optimizer,
            'lr_scheduler': scheduler,
            'monitor': 'loss'
        }


def main():
    cmd = parse_args()
    wandb.init(config=cmd, project="cifar10_vdp_resnet18")
    config = wandb.config
    model = LitResnet(psi=config.psi, tau=config.tau, wd=config.wd, pt=config.patience, optim=config.optim, sched=config.sched, lr=config.lr)
   
    wandb_logger = WandbLogger(log_model="all")
    checkpoint_callback = ModelCheckpoint(monitor="val_acc", mode="max")
    # early_stop_callback = EarlyStopping(monitor="loss", min_delta=0.00, patience=10, verbose=True, mode="min")
    trainer = Trainer(
        gpus=[6],
        max_epochs=210,
        accelerator=None,
        check_val_every_n_epoch=5,
        # devices=1 if torch.cuda.is_available() else None,  # limiting got iPython runs
        logger=wandb_logger,
        callbacks=[LearningRateMonitor(logging_interval="step"), checkpoint_callback] #, early_stop_callback],
    )

    trainer.fit(model, cifar10_dm)
    model = model.load_from_checkpoint(glob.glob(f'{wandb_logger.experiment.project}/{wandb_logger.experiment.id}/checkpoints/*.ckpt')[0])

    trainer.test(model, datamodule=cifar10_dm)

if __name__ =='__main__':
    main()