from EfficientNet.DET.efficientnet_pytorch import EfficientNet
from pytorch_lightning import LightningModule, Trainer, seed_everything
import PIL
import torchvision.transforms
import torch.nn as nn
import torch
import wandb
import argparse
from torchmetrics import Accuracy
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning import LightningModule, Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
import torch.nn.functional as F
from torchvision.transforms import transforms, AutoAugment, RandAugment
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import glob

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--lr', help='Learning Rate', type=float, default=0.1, required=False)
    parser.add_argument(
        '--wd', help='Weight Decay', type=float, default=0.0005, required=False)
    parser.add_argument(
        '--batch_size', help='Batch Size', type=int, default=128)
    parser.add_argument(
        '--patience', help='Patience', type=int, default=5)
    parser.add_argument(
        '--optim', help='Optimizer Type (adam, sgd)', type=str, default="adam")
    parser.add_argument(
        '--sched', help='Scheduler Type (multisteplr, cosine, plateau, onecycle)', type=str, default="cosine")
    parser.add_argument(
        '-a', '--arch', metavar='ARCH', default='efficientnet-b3',
                    help='model architecture (default: efficientnet-b0)')
    args = parser.parse_args()
    return args


def create_model(arch):
    model = EfficientNet.from_name(arch)
    return model

class LitResnet(LightningModule):
    def __init__(self, arch, wd=0.0005, pt=10, optim='adam', sched='onecycle', lr=0.0, batch_size=128):
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
        self.model = create_model(arch)
        self.criterion = torch.nn.CrossEntropyLoss()
        self.batch_size = batch_size

    @torch.enable_grad()
    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        self.train_acc(logits, y)
        self.log('loss', loss)
        self.log('train_acc', self.train_acc, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    @torch.enable_grad()
    def forward(self, x):
        out = self.model(x)
        # return F.log_softmax(out, dim=1)
        return out

    @torch.enable_grad()
    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        self.log('val_loss', loss)
        self.val_acc(logits, y)
        self.log('val_acc', self.val_acc, on_step=True, on_epoch=True, prog_bar=True)

    @torch.enable_grad()
    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        self.log('test_loss', loss)
        self.test_acc(logits, y)
        self.log('test_acc', self.test_acc, on_step=True, on_epoch=True)

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

    def train_dataloader(self):
        # transform = transforms.Compose([
        #     # AutoAugment(policy=torchvision.transforms.AutoAugmentPolicy.IMAGENET),
        #     transforms.Resize(256),
        #     transforms.CenterCrop(224),
        #     transforms.ToTensor(),
        #     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        # ])
        transform = transforms.Compose([
                transforms.Resize(256),
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
        train = ImageFolder('/data/tiny-imagenet-200/train', transform=transform)
        return DataLoader(train, batch_size=self.batch_size, num_workers=8, shuffle=True, pin_memory=True)

    def val_dataloader(self):
        # transform = transforms.Compose([
        #     transforms.Resize(256),
        #     transforms.CenterCrop(224),
        #     transforms.ToTensor(),
        #     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        # ])
        transform = transforms.Compose([
                transforms.Resize(256),
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
        test = ImageFolder('/data/tiny-imagenet-200/val', transform=transform)
        return DataLoader(test, batch_size=self.batch_size, num_workers=8, shuffle=False, pin_memory=True)

    def test_dataloader(self):
        # transform = transforms.Compose([
        #     transforms.Resize(256),
        #     transforms.CenterCrop(224),
        #     transforms.ToTensor(),
        #     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        # ])
        transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])  
        test = ImageFolder('/data/tiny-imagenet-200/test', transform=transform)
        return DataLoader(test, batch_size=self.batch_size, num_workers=8, shuffle=False, pin_memory=True)


def main():
    cmd = parse_args()
    wandb.init(config=cmd, project="tinyImageNet_efficientNet_vdp")
    config = wandb.config
    model = LitResnet(arch=config.arch, wd=config.wd, pt=config.patience, optim=config.optim, sched=config.sched, lr=config.lr, batch_size=config.batch_size)
   
    wandb_logger = WandbLogger(log_model="all")
    checkpoint_callback = ModelCheckpoint(monitor="val_acc", mode="max")
    # early_stop_callback = EarlyStopping(monitor="loss", min_delta=0.00, patience=10, verbose=True, mode="min")
    trainer = Trainer(
        gpus=[5],
        max_epochs=300,
        accelerator=None,
        check_val_every_n_epoch=5,
        # devices=1 if torch.cuda.is_available() else None,  # limiting got iPython runs
        logger=wandb_logger,
        callbacks=[LearningRateMonitor(logging_interval="step"), checkpoint_callback] #, early_stop_callback],
    )

    model.train()
    trainer.fit(model)
    model = model.load_from_checkpoint(glob.glob(f'{wandb_logger.experiment.project}/{wandb_logger.experiment.id}/checkpoints/*.ckpt')[0])

    trainer.test(model)

if __name__ =='__main__':
    main()