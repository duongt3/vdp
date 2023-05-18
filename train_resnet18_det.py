
import string
import numpy as np
import torchvision
from torchvision import *
import time

from torch.utils.data import DataLoader
import pandas as pd

from captum.attr import Saliency
from captum.metrics import infidelity,  sensitivity_max

import argparse

import matplotlib.pyplot as plt

import seaborn as sns
import scipy
from pl_bolts.datamodules import CIFAR10DataModule
from pl_bolts.transforms.dataset_normalizations import cifar10_normalization
from ResNet.ResNet18_DET_lightning import det_cifar10
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import WandbLogger
import wandb
from pytorch_lightning. callbacks import ModelCheckpoint
from datetime import datetime
# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"]="5"

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
        '--weight_decay', help='Weight Decay', type=float, default=0.0005, required=False)
    parser.add_argument(
        '--no_epochs', help='Number of Epochs', type=int, default=100, required=False)
    parser.add_argument(
        '--optim', help='Optimizer Type (adam, sgd)', type=str, default="sgd")
    parser.add_argument(
        '--sched', help='Scheduler Type (multisteplr, cosine, plateau, onecycle)', type=str, default="onecycle")
    parser.add_argument(
        '--patience', help='Number of Epochs until sched decreases', type=int, default=5, required=False)
    parser.add_argument(
        '--batch_size', help='Batch Size', type=int, default=128)
    parser.add_argument(
        '--num_runs', help="Number of models to train", type=int, default=1)
    args = parser.parse_args()
    return args

def train(model, epochs, save_dir, loggerIn):
    # wandb_logger = WandbLogger()
    start = time.time()
    checkpoint_callback = ModelCheckpoint(monitor="val_acc", mode="max")
    early_stop_callback = EarlyStopping(monitor="loss", min_delta=0.00, patience=20, verbose=False, mode="min")
    trainer = pl.Trainer(gpus=[5], max_epochs=epochs, check_val_every_n_epoch=5,  #auto_scale_batch_size='power',
                         accelerator=None, callbacks=[early_stop_callback, checkpoint_callback], logger=loggerIn)
    trainer.tune(model)
    # trainer.fit(model)
    # out = trainer.test(model)

    trainer.fit(model, cifar10_dm)
    out = trainer.test(model, datamodule=cifar10_dm)
    print("Total Train Time " + str(time.time()-start))
    test_acc = out[0]['test_acc_epoch']
    torch.save(model.state_dict(), save_dir)
    print("Saving " + save_dir)


def main():
    num_runs = 1
    cmd = parse_args()

    wandb.init(config=cmd, project="cifar10_det_resnet18")
    config = wandb.config

    model = det_cifar10(config.lr, config.weight_decay, config.patience, config.optim, config.sched, config.batch_size)
    dir = 'models/cifar_models/det_ResNet_' + str(config.num_runs) + ".pt"
    wandb_logger = WandbLogger(log_model="all")

    train(model, config.no_epochs, dir, wandb_logger)

if __name__ =='__main__':
    main()


