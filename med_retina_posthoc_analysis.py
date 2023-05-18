import os
import utils.vdp as vdp
import numpy as np
import torch
import pandas as pd
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
from captum.attr import Saliency
from torchvision import transforms
from collections import defaultdict
from LeNet.BBB import config_bayesian
from torch.utils.data import DataLoader
from scipy.stats import f_oneway, spearmanr
from captum.metrics import infidelity,  sensitivity_max
from pytorch_lightning import LightningModule, Trainer, seed_everything
from torch.nn.utils import prune

import ResNet.ResNet18_DET_lightning as resnet_det

from enum import Enum
import torchvision
import glob
from MedMnistDataModule import MedMnistDataModule
from medmnist.evaluator import getAUC, getACC

from train_resnet18_vdp_med import LitResnet as VDPModel
from train_resnet18_det_med import LitResnet as DetModel

vdp_checkpoint_dirs = ['0of8a9tc', '9fo0cpku','33khztx2', '67rvcwvs', 'mo0wsj5y']
det_checkpoint_dirs = ['5pruz3iq', 'f1knaufx', 'h6accqkk', 'l22llvay', 'rmif9hnm']

GPU = [0]

class ModelType(Enum):
    VDP = 1
    DET = 2


def get_model(model_type):
    if model_type == ModelType.VDP:
        model = VDPModel()
        model.to('cuda:0')
    elif model_type == ModelType.DET:
        model = DetModel()
        model.to('cuda:0')

    return model

def eval_test_set():
    metrics = list()
    models_to_evaluate = [ModelType.VDP, ModelType.DET]

    def add_metrics(model_type, checkpoint, test_acc, test_auc):
        metrics.append({'Model Type' : model_type,
                        'Checkpoint' : checkpoint,
                        'Test Accuracy' : test_acc,
                        'Test AUC' : test_auc})

    for model_type in models_to_evaluate:
        trainer = Trainer(gpus=GPU)
        model = get_model(model_type)

        if model_type == ModelType.VDP:
            for checkpoint in vdp_checkpoint_dirs:
                model = model.load_from_checkpoint(glob.glob(f'./models/med_vdp/{checkpoint}/checkpoints/*.ckpt')[0])
                result = trainer.test(model, datamodule=MedMnistDataModule())
                add_metrics(model_type.name, checkpoint, result[0]['test_acc'], result[0]['test_auc'])

        elif model_type == ModelType.DET:
            for checkpoint in det_checkpoint_dirs:
                model = model.load_from_checkpoint(glob.glob(f'./models/med_det/{checkpoint}/checkpoints/*.ckpt')[0])
                result = trainer.test(model, datamodule=MedMnistDataModule())
                add_metrics(model_type.name, checkpoint, result[0]['test_acc'], result[0]['test_auc'])

    if not os.path.isdir('experimental_results'):
        os.makedirs('experimental_results')
    pd.DataFrame(metrics).to_csv('experimental_results/retina_test_results.csv', index=False)

def statistical_analysis():
    test_df = pd.read_csv('experimental_results/retina_test_results.csv')

    vdp_test_accuracy = test_df.loc[test_df['Model Type'] == ModelType.VDP.name, 'Test Accuracy'].tolist()
    det_test_accuracy = test_df.loc[test_df['Model Type'] == ModelType.DET.name, 'Test Accuracy'].tolist()

    print('Test statistical analysis')
    print(f_oneway(vdp_test_accuracy, det_test_accuracy))

def compute_sensitivity_max(model, test_loader):

    ig = Saliency(model.to('cuda:0'))

    score = list()
    for itr, (x, labels) in enumerate(test_loader):
        attribution = ig.attribute(x.float().to('cuda:0'), target=labels.to('cuda:0'))
        score.append(sensitivity_max(ig.attribute, x.float().to('cuda:0'), perturb_radius=0.1, target=labels.to('cuda:0'), n_perturb_samples=5).detach().cpu().numpy())

    return np.mean(np.concatenate(score))


def compute_infidelity_score(model, test_loader):
    # ImageClassifier takes a single input tensor of images Nx3x32x32,
    # and returns an Nx10 tensor of class probabilities.
    saliency = Saliency(model.to('cuda:0'))
    score = list()

    for itr, (x, labels) in enumerate(test_loader):
        # Computes saliency maps for class 3.
        attribution = saliency.attribute(x.float().to('cuda:0'), target=labels.to('cuda:0'))
        # define a perturbation function for the input
        def perturb_fn(inputs):
            noise = torch.tensor(np.random.uniform(-0.5, 0.5, inputs.shape)).float().to('cuda:0')

            return noise, inputs - noise
        # Computes infidelity score for saliency maps
        score.append(infidelity(model, perturb_fn, x.float().to('cuda:0'), attribution, target=labels.to('cuda:0'), n_perturb_samples=5).detach().cpu().numpy())

    return np.mean(np.concatenate(score))

def add_noise(s, snr):
    var_s = np.var(s, axis=1)
    var_n = var_s / (10 ** (snr / 10))
    rand_arr = np.random.randn(s.shape[0], s.shape[1])
    n = np.sqrt(var_n).reshape((-1, 1)) * rand_arr
    return s + n

def snr_vs_gaussian_noise():
    models_to_evaluate = [ModelType.VDP]
    snrs = [-6, -3, 1, 5, 10, 20, 40]

    results = list()
    data_module = MedMnistDataModule()
    data_module.setup()
    testloader = data_module.test_dataloader()
    for model_type in models_to_evaluate:
        model = get_model(model_type)

        if model_type == ModelType.VDP:
            for checkpoint in vdp_checkpoint_dirs:
                model = model.load_from_checkpoint(glob.glob(f'./models/med_vdp/{checkpoint}/checkpoints/*.ckpt')[0])
                for snr in tqdm(snrs):
                    inner_sigmas = list()
                    for _, (image, _) in enumerate(testloader):
                        cur_batch_size = image.shape[0]
                        image = image.reshape(cur_batch_size, -1)
                        image = add_noise(image.numpy(), snr).reshape(cur_batch_size, 3, 28, 28)
                        mu, sigma = model.forward(torch.from_numpy(image).float())
                        preds = torch.argmax(mu, dim=1).detach().cpu().numpy()
                        sigma = sigma.detach().cpu().numpy()
                        uncertain = [sig[pred] for (pred, sig) in zip(preds, sigma)]
                        inner_sigmas.append(uncertain)
                    inner_sigmas = np.hstack(inner_sigmas)
                    results.append({'Model Type': model_type,
                                    'Checkpoint': checkpoint,
                                    'SNR (dB)': snr,
                                    'Mean Sigma': np.mean(inner_sigmas)})
                    pd.DataFrame(results).to_csv('experimental_results/retina_snr.csv', index=False)


def prune_vdp_parameters(model, percentage, save_directory):

    # These are the parameters for ResNet18, if pruning diff resnet need diff parameters
    parameter = ((model.conv1.mu, 'weight'),
                (model.layer1[0].conv1.mu, 'weight'),
                (model.layer1[0].conv2.mu, 'weight'),
                (model.layer1[1].conv1.mu, 'weight'),
                (model.layer1[1].conv2.mu, 'weight'),
                (model.layer2[0].conv1.mu, 'weight'),
                (model.layer2[0].conv2.mu, 'weight'),
                (model.layer2[1].conv1.mu, 'weight'),
                (model.layer2[1].conv2.mu, 'weight'),
                (model.layer3[0].conv1.mu, 'weight'),
                (model.layer3[0].conv2.mu, 'weight'),
                (model.layer3[1].conv1.mu, 'weight'),
                (model.layer3[1].conv2.mu, 'weight'),
                (model.layer4[0].conv1.mu, 'weight'),
                (model.layer4[0].conv2.mu, 'weight'),
                (model.layer4[1].conv1.mu, 'weight'),
                (model.layer4[1].conv2.mu, 'weight'),
                (model.fc.mu, 'weight'))

    prune.global_unstructured(
        parameter,
        pruning_method=prune.L1Unstructured,
        amount=percentage)

    for module, param in parameter:
        prune.remove(module, param)

    torch.save(model.state_dict(), save_directory)
    
    return model

def prune_det_parameters(model, percentage, save_directory):
    # These are the parameters for ResNet18, if pruning diff resnet need diff parameters
    parameter = ((model.conv1, 'weight'),
                (model.layer1[0].conv1, 'weight'),
                (model.layer1[0].conv2, 'weight'),
                (model.layer1[1].conv1, 'weight'),
                (model.layer1[1].conv2, 'weight'),
                (model.layer2[0].conv1, 'weight'),
                (model.layer2[0].conv2, 'weight'),
                (model.layer2[1].conv1, 'weight'),
                (model.layer2[1].conv2, 'weight'),
                (model.layer3[0].conv1, 'weight'),
                (model.layer3[0].conv2, 'weight'),
                (model.layer3[1].conv1, 'weight'),
                (model.layer3[1].conv2, 'weight'),
                (model.layer4[0].conv1, 'weight'),
                (model.layer4[0].conv2, 'weight'),
                (model.layer4[1].conv1, 'weight'),
                (model.layer4[1].conv2, 'weight'),
                (model.fc, 'weight'))
    
    prune.global_unstructured(
        parameter,
        pruning_method=prune.L1Unstructured,
        amount=percentage)

    for module, param in parameter:
        prune.remove(module, param)

    torch.save(model.state_dict(), save_directory)

    return model

def plot_snr():
    sns.set(font_scale=2)
    snr_df = pd.read_csv('experimental_results/retina_snr.csv')
    plt.figure(figsize=(18, 10))

    for model_type in snr_df['Model Type'].unique():
            for checkpoint in snr_df['Checkpoint'].unique():
                filter = (snr_df['Model Type'] == model_type) & (snr_df['Checkpoint'] == checkpoint)
                snr_df.loc[filter, 'Mean Sigma'] = snr_df.loc[filter]['Mean Sigma'] / snr_df.loc[filter]['Mean Sigma'].iloc[0]
    ax = sns.lineplot(x='SNR (dB)', y='Mean Sigma', data=snr_df, hue='Model Type', legend=True, marker='o', linewidth=3, markersize=10, palette=[sns.color_palette()[3]])
    # plt.legend(title='Model Type', labels=['VDP'])
    legend_labels, _= ax.get_legend_handles_labels()
    ax.legend(legend_labels, ['VDP++'])
    plt.ylabel('Normalized Mean Sigma')
    plt.savefig('results/retina/retina_snr.png')
    plt.clf()

def compute_sensitivity_max(model, test_loader):

    ig = Saliency(model.to('cuda:0'))

    score = list()
    for itr, (x, labels) in enumerate(test_loader):
        attribution = ig.attribute(x.float().to('cuda:0'), target=labels.to('cuda:0'))
        score.append(sensitivity_max(ig.attribute, x.float().to('cuda:0'), perturb_radius=0.1, target=labels.to('cuda:0'), n_perturb_samples=5).detach().cpu().numpy())

    return np.mean(np.concatenate(score))


def compute_infidelity_score(model, test_loader):
    # ImageClassifier takes a single input tensor of images Nx3x32x32,
    # and returns an Nx10 tensor of class probabilities.
    saliency = Saliency(model.to('cuda:0'))
    score = list()

    for itr, (x, labels) in enumerate(test_loader):
        # Computes saliency maps for class 3.
        attribution = saliency.attribute(x.float().to('cuda:0'), target=labels.to('cuda:0'))
        # define a perturbation function for the input
        def perturb_fn(inputs):
            noise = torch.tensor(np.random.uniform(-0.5, 0.5, inputs.shape)).float().to('cuda:0')

            return noise, inputs - noise
        # Computes infidelity score for saliency maps
        score.append(infidelity(model, perturb_fn, x.float().to('cuda:0'), attribution, target=labels.to('cuda:0'), n_perturb_samples=5).detach().cpu().numpy())

    return np.mean(np.concatenate(score))

if __name__ == '__main__':
    # eval_test_set()
    # snr_vs_gaussian_noise()
    # statistical_analysis()
    
    # plot_snr()
    # plot_prune()
    # plot_explainability()

    # prune_models()
    # explainability_analysis()
