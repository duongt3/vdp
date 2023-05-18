import argparse
import wandb
import torch
import glob
import numpy as np
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
from MedMnistDataModule import MedMnistDataModule
from medmnist.evaluator import getAUC, getACC

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
        '--sched', help='Scheduler Type (multisteplr, cosine, plateau, onecycle)', type=str, default="cosine")
    parser.add_argument(
        '--patience', help='Number of Epochs until sched decreases', type=int, default=10, required=False)
    parser.add_argument(
        '--batch_size', help='Batch Size', type=int, default=128)
    # parser.add_argument(
    #     '--num_runs', help="Number of models to train", type=int, default=1)
    parser.add_argument(
        '--psi', help="Hyperparameter for scaling nll loss term", type=int, default=700)
    parser.add_argument(
        '--tau', help="Hyperparameter for tuning KL loss term", type=float, default=0.01)
    args = parser.parse_args()
    return args


def create_model():
    model = ResNet_vdp.resnet18(pretrained=False, num_classes=5)
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
        y = torch.squeeze(y)
        mu, sigma = self(x)
        loss = self.get_loss(mu, sigma, y)
        self.log("loss", loss)
        # self.train_acc(mu.softmax(dim=-1), y)
        acc = getACC(y.detach().cpu().numpy(), mu.sigmoid().detach().cpu().numpy(), "lol")
        self.log('train_acc', acc, on_step=True, on_epoch=True, prog_bar=True)
        # self.log('train_auc', getAUC(np.asarray(list(map(convert_to_numeric, y.detach().cpu().numpy()))), mu.sigmoid(), "lol"), on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_auc', getAUC(y.detach().cpu().numpy(), mu.sigmoid().detach().cpu().numpy(), "lol" ), on_step=True, on_epoch=True, prog_bar=True)
        return loss

    @torch.enable_grad()
    def forward(self, x):
        mu, sigma = self.model(x)
        return mu, sigma

    def get_loss(self, mu, sigma, y):
        log_det, nll = vdp.ELBOLoss(mu, sigma, y, classes=5, multi_target=False)
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
        y = torch.squeeze(y)
        mu, sigma = self(x)
        loss = self.get_loss(mu, sigma, y)
        # acc = accuracy(mu.softmax(dim=-1), y)
        acc = getACC(y.detach().cpu().numpy(), mu.sigmoid().detach().cpu().numpy(), "lol")
        # auc = getAUC(np.asarray(list(map(convert_to_numeric, y.detach().cpu().numpy()))), mu.sigmoid().detach().cpu().numpy(), "lol")
        auc = getAUC(y.detach().cpu().numpy(), mu.sigmoid().detach().cpu().numpy(), "lol" )
        if stage:
            self.log(f"{stage}_loss", loss, prog_bar=True)
            self.log(f"{stage}_acc", acc, prog_bar=True)
            self.log(f"{stage}_auc", auc, prog_bar=True)

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
        else:
            raise ValueError('Invalid scheduler: ' + self.sched)

        return {
            'optimizer': optimizer,
            'lr_scheduler': scheduler,
            'monitor': 'loss'
        }

def convert_to_numeric(ordinal_arr):
    if np.array_equal(ordinal_arr, [1, 0, 0, 0, 0]):
        return 0
    elif np.array_equal(ordinal_arr, [1, 1, 0, 0, 0]):
        return 1
    elif np.array_equal(ordinal_arr, [1, 1, 1, 0, 0]):
        return 2
    elif np.array_equal(ordinal_arr, [1, 1, 1, 1, 0]):
        return 3
    elif np.array_equal(ordinal_arr, [1, 1, 1, 1, 1]):
        return 4

def main():
    cmd = parse_args()
    wandb.init(config=cmd, project="med_hard_vdp_resnet18")
    config = wandb.config
    model = LitResnet(psi=config.psi, tau=config.tau, wd=config.wd, pt=config.patience, optim=config.optim, sched=config.sched, lr=config.lr)

    # model = LitResnet(psi=774, tau=0.01703021444779236, wd=0.0005570343237797592, pt=config.patience, optim="adam", sched="plateau", lr=0.01947599293444144)
    wandb_logger = WandbLogger(log_model="all")
    checkpoint_callback = ModelCheckpoint(monitor="val_acc", mode="max")
    early_stop_callback = EarlyStopping(monitor="loss", min_delta=1, patience=3, verbose=True, mode="min")
    trainer = Trainer(
        gpus=[5],
        max_epochs=300,
        accelerator=None,
        check_val_every_n_epoch=5,
        # devices=1 if torch.cuda.is_available() else None,  # limiting got iPython runs
        logger=wandb_logger,
        callbacks=[LearningRateMonitor(logging_interval="step"), checkpoint_callback, early_stop_callback],
    )

    medMnist = MedMnistDataModule(batch_size=config.batch_size)

    trainer.fit(model, medMnist)
    model = model.load_from_checkpoint(glob.glob(f'{wandb_logger.experiment.project}/{wandb_logger.experiment.id}/checkpoints/*.ckpt')[0])
    trainer.test(model, datamodule=medMnist)

if __name__ =='__main__':
    main()