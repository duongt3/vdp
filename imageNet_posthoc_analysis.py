import os
import vdp
import time
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
from captum.attr import Saliency
from torchvision import transforms
from collections import defaultdict
from torch.utils.data import DataLoader
from scipy.stats import f_oneway, spearmanr
from captum.metrics import infidelity, sensitivity_max
from torch.nn.utils import prune
from torchvision.datasets import ImageFolder
import pytorch_lightning as pl


from enum import Enum
import torchvision
import glob

from ViT.helper import prune_vdp_parameters, prune_det_parameters
from ViT.ViT_lightning import ViT
from ViT.train_ImageNet_vdp import ViT_vdp
from torchvision.models import vit_b_16, ViT_B_16_Weights

class Config:
    pass
config = Config()
config.alpha = 5729
config.lr = 0.006131644
config.optim = "adam"
config.sched = "CyclicLR"
config.tau = 0.0008034
config.wd = 0.0008966
config.batch_size = 1024
GPU=[5]

class ModelType(Enum):
    VDP = 1
    DET = 2

def test_dataloader(batch_size):
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    test = ImageFolder('/data/ImageNet/val', transform=transform)
    return DataLoader(test, batch_size=batch_size, num_workers=8, shuffle=False, pin_memory=True)

def get_model(model_type):
    if model_type == ModelType.VDP:
        model = ViT_vdp(config)
        model = model.load_from_checkpoint('lightning_logs/3mdxxyd3/checkpoints/epoch=299-step=62700.ckpt', config=config, num_classes=1000)
        # model.to('cuda:0')
    elif model_type == ModelType.DET:
        model = ViT(vit_b_16(weights=ViT_B_16_Weights.DEFAULT))
        # model.to('cuda:0')

    return model

def eval_test_set():
    metrics = list()
    models_to_evaluate = [ ModelType.VDP, ModelType.DET]

    def add_metrics(model_type, test_acc):
        metrics.append({'Model Type' : model_type,
                        'Test Accuracy' : test_acc})

    for model_type in models_to_evaluate:
        trainer = pl.Trainer(gpus=[0], strategy='ddp', max_epochs=1, check_val_every_n_epoch=5, #auto_scale_batch_size='power',
                            accelerator='gpu', inference_mode=False)
        model = get_model(model_type)
        trainer.tune(model)

        if model_type == ModelType.VDP:
            result = trainer.test(model) #Changed the dataloader to train set
            add_metrics(model_type.name, result[0]['test_acc'])

        if model_type == ModelType.DET:
            result = trainer.test(model)
            add_metrics(model_type.name, result[0]['test_acc'])

    if not os.path.isdir('experimental_results'):
        os.makedirs('experimental_results')

    pd.DataFrame(metrics).to_csv('experimental_results/imagenet_test_results.csv', index=False)


def add_noise(s, snr):
    var_s = np.var(s, axis=1)
    var_n = var_s / (10 ** (snr / 10))
    rand_arr = np.random.randn(s.shape[0], s.shape[1])
    n = np.sqrt(var_n).reshape((-1, 1)) * rand_arr
    return s + n

def snr_vs_gaussian_noise():
    snrs = [-6, -3, 1, 5, 10, 20, 40]
    results = list()
    
    model = get_model(ModelType.VDP)

    testloader = test_dataloader()
    for snr in snrs:
        inner_sigmas = list()
        for i, (image, _) in enumerate(testloader):
            print(i)
            cur_batch_size = image.shape[0]
            image = image.reshape(cur_batch_size, -1)
            image = add_noise(image.numpy(), snr).reshape(cur_batch_size, 3, 224, 224)
            mu, sigma = model.forward(torch.from_numpy(image).float())
            preds = torch.argmax(mu, dim=1).detach().cpu().numpy()
            sigma = sigma.detach().cpu().numpy()
            uncertain = [sig[pred] for (pred, sig) in zip(preds, sigma)]
            inner_sigmas.append(uncertain)
        inner_sigmas = np.hstack(inner_sigmas)
        results.append({'Model Type': "VDP",
                        'SNR (dB)': snr,
                        'Mean Sigma': np.mean(inner_sigmas)})
        pd.DataFrame(results).to_csv('experimental_results/imageNet_snr.csv', index=False)

def prune_models():
    metrics= list()
    prune_percentages = [0.25, 0.5, 0.75, 0.85, 0.9, 0.95, 0.975, 0.99]
    def add_metrics(model_type, test_acc, prune_percentage):
        metrics.append({'Model Type' : model_type,
                        'Test Accuracy' : test_acc,
                        'Prune Percentage' : prune_percentage})


    trainer = pl.Trainer(gpus=[0], strategy='ddp', check_val_every_n_epoch=5, #auto_scale_batch_size='power',
                        accelerator='gpu', inference_mode=False)
    try:
        # for percentage in prune_percentages:
        #     model = get_model(ModelType.DET)
        #     directory = 'prunedModels/ImageNet/DET/model_' + str(percentage) + '.ct'
        #     prune_det_parameters(model, percentage, directory)
        #     result = trainer.test(model)
        #     add_metrics(ModelType.DET.name, result[0]['test_acc'], percentage)

        for percentage in prune_percentages:
            model = get_model(ModelType.VDP)
            directory = 'prunedModels/ImageNet/VDP/model_' + str(percentage) + '.ct'
            prune_vdp_parameters(model, percentage, directory)
            result = trainer.test(model)
            add_metrics(ModelType.VDP.name, result[0]['test_acc'], percentage)
    finally:
        if not os.path.isdir('experimental_results'):
            os.makedirs('experimental_results')
        pd.DataFrame(metrics).to_csv('experimental_results/imageNet_pruning_results.csv', index=False)

def explainability_analysis():
    def compute_sensitivity_max(model, test_loader):

        ig = Saliency(model.to('cuda:0'))

        score = list()
        for itr, (x, labels) in enumerate(test_loader):
            print("Sensitivity iteration: " + str(itr))
            attribution = ig.attribute(x.float().to('cuda:0'), target=labels.to('cuda:0'))
            score.append(sensitivity_max(ig.attribute, x.float().to('cuda:0'), perturb_radius=0.1, target=labels.to('cuda:0'), n_perturb_samples=5).detach().cpu().numpy())

        return np.mean(np.concatenate(score))

    def compute_infidelity_score(model, test_loader):
        # ImageClassifier takes a single input tensor of images Nx3x32x32,
        # and returns an Nx10 tensor of class probabilities.
        saliency = Saliency(model.to('cuda:0'))
        score = list()

        for itr, (x, labels) in enumerate(test_loader):
            print("Infidelity iteration: " + str(itr))
            # Computes saliency maps for class 3.
            attribution = saliency.attribute(x.float().to('cuda:0'), target=labels.to('cuda:0'))
            # define a perturbation function for the input
            def perturb_fn(inputs):
                noise = torch.tensor(np.random.uniform(-0.5, 0.5, inputs.shape)).float().to('cuda:0')

                return noise, inputs - noise
            # Computes infidelity score for saliency maps
            score.append(infidelity(model, perturb_fn, x.float().to('cuda:0'), attribution, target=labels.to('cuda:0'), n_perturb_samples=5).detach().cpu().numpy())

        return np.mean(np.concatenate(score))
    
    models_to_evaluate = [ModelType.VDP, ModelType.DET]
    scores = list()
    try:
        for model_type in models_to_evaluate:
            model = get_model(model_type)
            if model_type == ModelType.VDP:
                dataloader = test_dataloader(16)
                inf = compute_infidelity_score(model, dataloader)
                sensitivity = compute_sensitivity_max(model, dataloader)
                scores.append({'Model Type': model_type, 'Sensitivity Max':sensitivity, 'Infidelity': inf})
                pd.DataFrame(scores).to_csv('experimental_results/imageNet_explainability_vdp.csv', index=False)
            elif model_type == ModelType.DET:
                dataloader = test_dataloader(8)
                sensitivity = compute_sensitivity_max(model.model, dataloader)
                inf = compute_infidelity_score(model.model, dataloader)
                scores.append({'Model Type': model_type, 'Sensitivity Max':sensitivity, 'Infidelity': inf})
                pd.DataFrame(scores).to_csv('experimental_results/imageNet_explainability_det.csv', index=False)
    finally:
        pd.DataFrame(scores).to_csv('experimental_results/imageNet_explainability.csv', index=False)

def plot_snr():
    sns.set(font_scale=2)
    snr_df = pd.read_csv('experimental_results/imageNet_snr.csv')
    plt.figure(figsize=(18, 10))
    snr_df['Mean Sigma']-= snr_df['Mean Sigma'].min()
    snr_df['Mean Sigma'] = snr_df['Mean Sigma'] / snr_df['Mean Sigma'].iloc[0]
    ax = sns.lineplot(x='SNR (dB)', y='Mean Sigma', data=snr_df, hue='Model Type', legend=True, marker='o', linewidth=3, markersize=10, palette=[sns.color_palette()[3]])
    # plt.legend(title='Model Type', labels=['VDP'])
    legend_labels, _= ax.get_legend_handles_labels()
    ax.legend(legend_labels, ['VDP++'])
    plt.ylabel('Normalized Mean Sigma')
    plt.savefig('results/imageNet_snr.png')
    plt.clf()

def plot_prune():
    sns.set(font_scale=2)
    pruning_df = pd.read_csv('experimental_results/imageNet_pruning_results.csv')

    plt.figure(figsize=(18, 10))
    ax = sns.lineplot(x='Prune Percentage', y='Accuracy', data=pruning_df, hue='Model Type', legend=True, marker='o', linewidth=3, markersize=10, palette=[sns.color_palette()[0], sns.color_palette()[3]])
    plt.xlabel('Percentage of Parameters Pruned')
    legend_labels, _= ax.get_legend_handles_labels()
    ax.legend(legend_labels, ['DET', 'VDP++'])
    # plt.legend(title='Model Type', labels=['DET', 'VDP++'])
    plt.savefig('results/imageNet_pruning.png')
    plt.clf()

def plot_explainability():
    plt.figure(figsize=(18, 10))
    sns.set(font_scale=2)
    explainability_df = pd.read_csv('experimental_results/imageNet_explainability.csv')
    fig, axes = plt.subplots(1, 2, figsize=(18, 10))
    sns.barplot(x='Model Type', y='Sensitivity Max', data=explainability_df, order=['ModelType.DET', 'ModelType.VDP'], ax=axes[0], palette=[sns.color_palette()[0], sns.color_palette()[3]])
    axes[0].set_xticklabels(['DET', 'VDP++'])
    # axes[0].set_yscale('log')
    sns.barplot(x='Model Type', y='Infidelity', data=explainability_df, ax=axes[1], order=['ModelType.DET', 'ModelType.VDP'], palette=[sns.color_palette()[0], sns.color_palette()[3]])
    axes[1].set_xticklabels(['DET', 'VDP++'])
    # axes[1].set_yscale('log')
    plt.savefig('results/imageNet_explainability.png')
    plt.clf()

if __name__ == '__main__':
    #eval_test_set() #done
    # snr_vs_gaussian_noise() #done
    # prune_models() #done
    # explainability_analysis()#in-progress

    # statistical_analysis()
    
    plot_snr() # done
    # plot_prune() #done
    # plot_explainability() #todo

    #test_acc_vs_gaussian_noise()#todo