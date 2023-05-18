import vdp
import torch
import wandb
import argparse
import torchvision
import pytorch_lightning as pl
from torchmetrics import Accuracy
from torch.utils.data import DataLoader
from simple_vit_vdp import SimpleViT_vdp
from torchvision.datasets import ImageFolder
from torchvision.transforms import transforms
from torchvision.transforms import RandAugment, AugMix 
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--lr', help='Learning Rate', type=float, default=1e-3, required=False)
    parser.add_argument(
        '--wd', help='Weight Decay', type=float, default=0, required=False)
    parser.add_argument(
        '--optim', help='Optimizer Type (adam, sgd)', type=str, default="adam")
    parser.add_argument(
        '--sched', help='Scheduler Type (multisteplr, cosine, reduceLROnPlateau, onecycle)', type=str, default="CyclicLR")
    parser.add_argument(
        '--batch_size', help='Batch Size', type=int, default=1024)
    parser.add_argument(
        '--alpha', help="Hyperparameter for scaling nll loss term", type=int, default=1000)
    parser.add_argument(
        '--tau', help="Hyperparameter for tuning KL loss term", type=float, default=0.01)
    parser.add_argument(
        '--local_rank', help='local rank of ddp process', type=int, default=-1
    )
    args = parser.parse_args()
    return args


class ViT_vdp(SimpleViT_vdp, pl.LightningModule):
    def __init__(self, config, num_classes=1000):
        super().__init__(image_size = 256, patch_size = 32, channels = 3, num_classes = num_classes, dim = 384, depth = 12, heads = 6, mlp_dim = 1536)
        self.lr = config.lr
        self.wd = config.wd
        self.alpha = config.alpha
        self.tau = config.tau
        self.batch_size = config.batch_size
        self.optim = config.optim
        self.sched = config.sched

        self.num_classes = num_classes
        self.train_acc = Accuracy(task='multiclass', num_classes=num_classes, top_k=1)
        self.val_acc = Accuracy(task='multiclass', num_classes=num_classes, top_k=1)
        self.test_acc = Accuracy(task='multiclass', num_classes=num_classes, top_k=1)

    def get_loss(self, mu, sigma, y):
        log_det, nll = vdp.ELBOLoss(mu, sigma, y, self.num_classes, criterion='ce')
        kl = vdp.gather_kl(self)
        # wandb.log({"log_det": log_det,
        #           "nll": self.alpha * nll,
        #           "kl": self.tau * sum(kl)})
        loss = log_det + self.alpha * nll + self.tau * sum(kl)
        # loss = log_det + 1000 * nll + sum(kl)
        return loss

    @torch.enable_grad()
    def training_step(self, batch, batch_idx):
        x, y = batch
        mu, sigma = self(x)
        loss = self.get_loss(mu, sigma, y)
        self.log('loss', loss, rank_zero_only=True)
        self.train_acc(mu.softmax(dim=-1), y)
        self.log('train_acc', self.train_acc, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True, rank_zero_only=True)
        return loss

    @torch.enable_grad()
    def on_train_epoch_end(self):
        self.log('train_acc', self.train_acc, rank_zero_only=True)

    @torch.enable_grad()
    def validation_step(self, batch, batch_idx):
        x, y = batch
        mu, sigma = self(x)
        loss = self.get_loss(mu, sigma, y)
        self.log('val_loss', loss, rank_zero_only=True)
        self.val_acc(mu.softmax(dim=-1), y)
        self.log('val_acc', self.val_acc, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True, rank_zero_only=True)

    @torch.enable_grad()
    def test_step(self, batch, batch_idx):
        x, y = batch
        mu, sigma = self(x)
        loss = self.get_loss(mu, sigma, y)
        self.log('test_loss', loss, rank_zero_only=True)
        self.test_acc(mu.softmax(dim=-1), y)
        self.log('test_acc', self.test_acc, on_step=True, on_epoch=True, sync_dist=True, rank_zero_only=True)

    @torch.enable_grad()
    def on_test_epoch_end(self):
        self.log('test_acc', self.test_acc, rank_zero_only=True)

    def train_dataloader(self):
        transform = transforms.Compose([
            # RandAugment(magnitude=10),
            # AugMix(),
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        train = ImageFolder('/data/ImageNet/train', transform=transform)
        return DataLoader(train, batch_size=self.batch_size, num_workers=8, shuffle=True, pin_memory=True)


    def val_dataloader(self):
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        val = ImageFolder('/data/ImageNet/val', transform=transform)
        return DataLoader(val, batch_size=self.batch_size, num_workers=8, shuffle=False, pin_memory=True)
    
    def test_dataloader(self):
        transform = transforms.Compose([
            # RandAugment(magnitude=10),
            # AugMix(),
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        train = ImageFolder('/data/ImageNet/train', transform=transform)
        return DataLoader(train, batch_size=self.batch_size, num_workers=8, shuffle=False, pin_memory=True)
    
    # def test_dataloader(self):
    #     transform = transforms.Compose([
    #         transforms.Resize(256),
    #         transforms.CenterCrop(224),
    #         transforms.ToTensor(),
    #         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    #     ])
    #     test = ImageFolder('/data/ImageNet/val', transform=transform)
    #     return DataLoader(test, batch_size=self.batch_size, num_workers=8, shuffle=False, pin_memory=True)

    def configure_optimizers(self):
        if self.optim == "adam":
            optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.wd, amsgrad=True)
        else:
            optimizer = torch.optim.SGD(self.parameters(), lr=self.lr,
                                        momentum=0.9, weight_decay=self.wd)

        if self.sched == "multisteplr":
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [10, 20, 30, 40, 50, 60, 70, 80, 90], gamma=0.5)
        elif self.sched == "reduceLROnPlateau":            
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2)
        elif self.sched == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
        elif self.sched == "CyclicLR":
            scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=self.lr, max_lr=self.lr*10, cycle_momentum=False)
        elif self.sched == "OneCycleLR":
            scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.01, steps_per_epoch=len(self.train_dataloader()), epochs=101)

        return {
            'optimizer': optimizer,
            'lr_scheduler': scheduler,
            'monitor': 'loss'
        }


def train(model, epochs, config):
    wandb_logger = WandbLogger()
    checkpoint_callback = ModelCheckpoint(save_weights_only=True)
    trainer = pl.Trainer(gpus=6, strategy='ddp', max_epochs=epochs, check_val_every_n_epoch=5, logger=wandb_logger, #auto_scale_batch_size='power',
                         accelerator='gpu', inference_mode=False, callbacks=[checkpoint_callback])
    trainer.tune(model)
    trainer.fit(model)
    out = trainer.test(model)
    print(out)
    wandb.finish()
    # test_acc = out[0]['test_acc_epoch']
    # torch.save(model.state_dict(), save_dir)
    # print("Saving " + save_dir)

if __name__ =='__main__':
    cmd = parse_args()
    if cmd.local_rank == 0:
        wandb.init(config=cmd, project="ViT_ImageNet")
        config = wandb.config
    else:
        config = cmd
    model = ViT_vdp(config) # 512 local
    model = model.load_from_checkpoint('lightning_logs/a4v3shwe/checkpoints/epoch=99-step=31300.ckpt', config=config, num_classes=1000)
    train(model, epochs=300, config=config)
