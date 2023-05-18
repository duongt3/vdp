import os
import utils.vdp as vdp
import utils.EVI as EVI
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
from LeNet.BBB import config_bayesian
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from scipy.stats import f_oneway, spearmanr
from captum.metrics import infidelity,  sensitivity_max
from pytorch_lightning import LightningModule, Trainer, seed_everything
from torch.nn.utils import prune

import ResNet.ResNet18_DET_lightning as resnet_det

import train_resnet18_vdp as resnet_vdp_cifar
import train_resnet18_vdp_med as resnet_vdp_pathMnist
from enum import Enum
from pl_bolts.datamodules import CIFAR10DataModule
from pl_bolts.transforms.dataset_normalizations import cifar10_normalization
import torchvision
import glob
from torchvision.datasets import CIFAR10

vdp_checkpoint_dirs = ['2kqf8ewj', '13kx3rx7', '231605qm', 'mndx8nli', 'w7vu4kt5']
vdp_finetune_dirs = ['tjf7brtm', '76idkahj', 'gty6p9in', 'p2t9xtnr', 'kzogzqsy']
det_checkpoint_dirs = ['2ekys6qo', '3lgv16g0', '2sntbkp9', '5hqmsi2i', 'ki55zx0f']

GPU = [0]

def get_datamodule():

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

    return cifar10_dm

class ModelType(Enum):
    VDPCIFAR10 = 1
    DETCIFAR10 = 2

def default_batch():
    return 512

def get_model(model_type):
    if model_type == ModelType.VDPCIFAR10:
        model = resnet_vdp_cifar.LitResnet()
        # model.to('cuda:0')
    elif model_type == ModelType.DETCIFAR10:
        model = resnet_det.det_cifar10()
        # model.to('cuda:0')

    return model

def eval_test_set():
    metrics = list()
    models_to_evaluate = [ ModelType.VDPCIFAR10, ModelType.DETCIFAR10]

    def add_metrics(model_type, checkpoint, test_acc, time):
        metrics.append({'Model Type' : model_type,
                        'Checkpoint' : checkpoint,
                        'Test Accuracy' : test_acc,
                        'Inference Time' : time})

    for model_type in models_to_evaluate:
        trainer = Trainer(gpus=GPU)
        model = get_model(model_type)

        if model_type == ModelType.VDPCIFAR10:
            for checkpoint in vdp_checkpoint_dirs:
                model = model.load_from_checkpoint(glob.glob(f'./cifar10_vdp_resnet18/{checkpoint}/checkpoints/*.ckpt')[0])

                start_time = time.time()
                result = trainer.test(model, datamodule=get_datamodule())
                add_metrics(model_type.name, checkpoint, result[0]['test_acc'], time.time()-start_time)

        elif model_type == ModelType.DETCIFAR10:
            for checkpoint in det_checkpoint_dirs:
                model = model.load_from_checkpoint(glob.glob(f'./cifar10_det_resnet18/{checkpoint}/checkpoints/*.ckpt')[0])

                start_time = time.time()
                result = trainer.test(model, datamodule=get_datamodule())
                add_metrics(model_type.name, checkpoint, result[0]['test_acc'], time.time()-start_time)

    if not os.path.isdir('experimental_results'):
        os.makedirs('experimental_results')
    pd.DataFrame(metrics).to_csv('experimental_results/cifar10_test_results.csv', index=False)

def add_noise(s, snr):
    var_s = np.var(s, axis=1)
    var_n = var_s / (10 ** (snr / 10))
    rand_arr = np.random.randn(s.shape[0], s.shape[1])
    n = np.sqrt(var_n).reshape((-1, 1)) * rand_arr
    return s + n

def test_acc_vs_gaussian_noise():
    models_to_evaluate = [ModelType.VDPCIFAR10, ModelType.DETCIFAR10]
    snrs = [-6, -3, 1, 5, 10, 20, 40]

    results = list()
    transforms = torchvision.transforms.Compose(
        [
            torchvision.transforms.ToTensor(),
            cifar10_normalization(),
        ]
    )
    test = CIFAR10(os.getcwd(), train=False, download=True, transform=transforms)
    testloader = DataLoader(test, batch_size=4096, num_workers=int(os.cpu_count() / 2), shuffle=True, pin_memory=True)
    for model_type in tqdm(models_to_evaluate):
        model = get_model(model_type)

        if model_type == ModelType.VDPCIFAR10:
            for checkpoint in tqdm(vdp_finetune_dirs):
                model = model.load_from_checkpoint(glob.glob(f'./cifar10_vdp_finetune/{checkpoint}/checkpoints/*.ckpt')[0])
                for snr in tqdm(snrs):
                    predicted = list()
                    truth = list()
                    for _, (image, labels) in tqdm(enumerate(testloader)):
                        cur_batch_size = image.shape[0]
                        image = image.reshape(cur_batch_size, -1)
                        image = add_noise(image.numpy(), snr).reshape(cur_batch_size, 3, 32, 32)
                        if model_type == ModelType.VDPCIFAR10:
                            pred, _ = model.forward(torch.from_numpy(image).float())
                        else:
                            pred = model.forward(torch.from_numpy(image).float())
                            
                        preds = torch.argmax(pred, dim=1).detach().cpu().numpy().tolist()
                        predicted.append(preds)
                        truth.append(labels.numpy().tolist())
                    predicted = np.hstack(predicted)
                    truth = np.hstack(truth)
                    test_acc = sum([p == t for (p, t) in zip(predicted, truth)]) / len(truth)
                    results.append({'Model Type': model_type,
                                    'Checkpoint': checkpoint,
                                    'SNR (dB)': snr,
                                    'Test Acc': test_acc})
                    pd.DataFrame(results).to_csv('results/cifar10/cifar10_snr_test_acc.csv', index=False)
        else:
            for checkpoint in tqdm(det_checkpoint_dirs):
                model = model.load_from_checkpoint(glob.glob(f'./cifar10_det_resnet18/{checkpoint}/checkpoints/*.ckpt')[0])
                for snr in tqdm(snrs):
                    predicted = list()
                    truth = list()
                    for _, (image, labels) in enumerate(testloader):
                        cur_batch_size = image.shape[0]
                        image = image.reshape(cur_batch_size, -1)
                        image = add_noise(image.numpy(), snr).reshape(cur_batch_size, 3, 32, 32)
                        if model_type == ModelType.VDPCIFAR10:
                            pred, _ = model.forward(torch.from_numpy(image).float())
                        else:
                            pred = model.forward(torch.from_numpy(image).float())
                            
                        preds = torch.argmax(pred, dim=1).detach().cpu().numpy().tolist()
                        predicted.append(preds)
                        truth.append(labels.numpy().tolist())
                    predicted = np.hstack(predicted)
                    truth = np.hstack(truth)
                    test_acc = sum([p == t for (p, t) in zip(predicted, truth)]) / len(truth)
                    results.append({'Model Type': model_type,
                                    'Checkpoint': checkpoint,
                                    'SNR (dB)': snr,
                                    'Test Acc': test_acc})
                    pd.DataFrame(results).to_csv('results/cifar10/cifar10_snr_test_acc.csv', index=False)
                    

def snr_vs_gaussian_noise():
    models_to_evaluate = [ModelType.VDPCIFAR10]
    snrs = [-6, -3, 1, 5, 10, 20, 40]
    results = list()
    transforms = torchvision.transforms.Compose(
        [
            torchvision.transforms.ToTensor(),
            cifar10_normalization(),
        ]
    )
    test = CIFAR10(os.getcwd(), train=False, download=True, transform=transforms)
    testloader = DataLoader(test, batch_size=default_batch(), num_workers=2, shuffle=True, pin_memory=True)
    for model_type in models_to_evaluate:
        model = get_model(model_type)

        if model_type == ModelType.VDPCIFAR10:
            for checkpoint in vdp_checkpoint_dirs:
                model = model.load_from_checkpoint(glob.glob(f'./cifar10_vdp_resnet18/{checkpoint}/checkpoints/*.ckpt')[0])
                for snr in tqdm(snrs):
                    inner_sigmas = list()
                    for _, (image, _) in enumerate(testloader):
                        cur_batch_size = image.shape[0]
                        image = image.reshape(cur_batch_size, -1)
                        image = add_noise(image.numpy(), snr).reshape(cur_batch_size, 3, 32, 32)
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
                    # pd.DataFrame(results).to_csv('experimental_results/cifar10_snr.csv', index=False)


def get_layers_vdp(model):
    layers = list()
    for name, layer in model.named_children():
        if type(layer) == vdp.Conv2d or type(layer) == vdp.Linear:
            mus = layer.mu.weight.reshape(-1, 1).detach().cpu().numpy().squeeze()
            sigmas = layer.sigma.weight.reshape(-1, 1).detach().cpu().numpy().squeeze()
            [layers.append({'Layer Name': name, 'Mu': mu}) for mu in mus]
            [layers.append({'Layer Name': name, 'Sigma': sigma}) for sigma in sigmas]
    return layers

def statistical_analysis():
    test_df = pd.read_csv('experimental_results/cifar10_test_results.csv')

    vdp_test_accuracy = test_df.loc[test_df['Model Type'] == ModelType.VDPCIFAR10.name, 'Test Accuracy'].tolist()
    det_test_accuracy = test_df.loc[test_df['Model Type'] == ModelType.DETCIFAR10.name, 'Test Accuracy'].tolist()

    print('Test statistical analysis')
    print(f_oneway(vdp_test_accuracy, det_test_accuracy))

    # snr_df = pd.read_csv('experimental_results/cifar10_snr.csv')
    # snr_df = snr_df.loc[snr_df['SNR (dB)'] != -6]

    # vdp_snr = snr_df.loc[snr_df['Model Type'] == ModelType.VDPCIFAR10.name, 'Mean Sigma']
    # vdp_snr = np.array([vdp_snr[i::5] for i in range(5)])
    # print('SNR statistical analysis')
    # print([spearmanr(vdp_snr[i]) for i in range(len(vdp_snr))])
    pass

def plot_snr():
    sns.set(font_scale=2)
    snr_df = pd.read_csv('results/cifar10/cifar10_snr_finetune.csv')
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
    plt.savefig('results/cifar10/cifar10_snr.png')
    plt.clf()

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

def prune_models():
    metrics= list()
    trainer = Trainer(gpus=GPU)
    prune_percentages = [0.25, 0.5, 0.75, 0.85, 0.9, 0.95, 0.975, 0.99, 0.999]
    def add_metrics(model_type, checkpoint, test_acc, prune_percentage):
        metrics.append({'Model Type' : model_type,
                        'Checkpoint' : checkpoint,
                        'Test Accuracy' : test_acc,
                        'Prune Percentage' : prune_percentage})

    for checkpoint in det_checkpoint_dirs:
        for percentage in prune_percentages:
            parameters = list()
            model = get_model(ModelType.DETCIFAR10)
            directory = 'pruneModels/DETResnet18/model_' + str(percentage) + '.ct'
            prune_det_parameters(model.model, percentage, directory)
            result = trainer.test(model, datamodule=get_datamodule())
            add_metrics(ModelType.DETCIFAR10.name, checkpoint, result[0]['test_acc'], percentage)

    for checkpoint in vdp_checkpoint_dirs:
        for percentage in prune_percentages:
            parameters = list()
            model = get_model(ModelType.VDPCIFAR10)
            directory = 'pruneModels/VDPResnet18/model_' + str(percentage) + '.ct'
            prune_vdp_parameters(model.model, percentage, directory)
            result = trainer.test(model, datamodule=get_datamodule())
            add_metrics(ModelType.VDPCIFAR10.name, checkpoint, result[0]['test_acc'], percentage)

    if not os.path.isdir('experimental_results'):
        os.makedirs('experimental_results')
    pd.DataFrame(metrics).to_csv('results/cifar10/cifar10_pruning_results.csv', index=False)
    
    
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


def explainability_analysis():
    models_to_evaluate = [ModelType.VDPCIFAR10, ModelType.DETCIFAR10]
    scores = list()
    transforms = torchvision.transforms.Compose(
        [
            torchvision.transforms.ToTensor(),
            cifar10_normalization(),
        ]
    )
    
    testset = torchvision.datasets.CIFAR10(root='cifar-10-batches-py', train=False,
                                        download=True, transform=transforms)
    testloader = torch.utils.data.DataLoader(testset, batch_size=256,
                                            shuffle=False, num_workers=2)
    for model_type in tqdm(models_to_evaluate):
        model = get_model(model_type)
        if model_type == ModelType.VDPCIFAR10:
            for checkpoint in tqdm(vdp_checkpoint_dirs):
                model = model.load_from_checkpoint(glob.glob(f'./cifar10_vdp_resnet18/{checkpoint}/checkpoints/*.ckpt')[0])
                model.to('cuda:0')
                sensitivity = compute_sensitivity_max(model, testloader)
                infidelity = compute_infidelity_score(model, testloader)
                scores.append({'Model Type': model_type, 'KF':checkpoint, 'Sensitivity Max':sensitivity, 'Infidelity': infidelity})
        elif model_type == ModelType.DETCIFAR10:
            for checkpoint in tqdm(det_checkpoint_dirs):
                model = model.load_from_checkpoint(glob.glob(f'./cifar10_det_resnet18/{checkpoint}/checkpoints/*.ckpt')[0])
                model.to('cuda:0')
                sensitivity = compute_sensitivity_max(model, testloader)
                infidelity = compute_infidelity_score(model, testloader)
                scores.append({'Model Type': model_type, 'KF':checkpoint, 'Sensitivity Max':sensitivity, 'Infidelity': infidelity})
        pd.DataFrame(scores).to_csv('results/cifar10/cifar10_explainability.csv', index=False)


def plot_prune():
    sns.set(font_scale=2)
    pruning_df = pd.read_csv('results/cifar10/cifar10_pruning_results.csv')
    all_prune = list([
        {'Model Type': 'DETCIFAR10', '3lgv16g0': 0, 'Prune Percentage': 1, 'Test Accuracy': 0},
        {'Model Type': 'VDPCIFAR10', 'w7vu4kt5': 0, 'Prune Percentage': 1, 'Test Accuracy': 0},      
    ])
    pruning_df = pd.concat([pruning_df, pd.DataFrame(all_prune)], ignore_index=True)
    plt.figure(figsize=(18, 10))
    ax = sns.lineplot(x='Prune Percentage', y='Test Accuracy', data=pruning_df, hue='Model Type', legend=True, marker='o', linewidth=3, markersize=10, palette=[sns.color_palette()[0], sns.color_palette()[3]])
    plt.xlabel('Percentage of Parameters Pruned')
    legend_labels, _= ax.get_legend_handles_labels()
    ax.legend(legend_labels, ['DET', 'VDP++'])
    # plt.legend(title='Model Type', labels=['DET', 'VDP++'])
    plt.savefig('results/cifar10/cifar10_pruning.png')
    plt.clf()
    
    
def plot_explainability():
    plt.figure(figsize=(18, 10))
    sns.set(font_scale=2)
    explainability_df = pd.read_csv('results/cifar10/cifar10_explainability.csv')
    fig, axes = plt.subplots(1, 2, figsize=(18, 10))
    sns.barplot(x='Model Type', y='Sensitivity Max', data=explainability_df, order=['ModelType.DETCIFAR10', 'ModelType.VDPCIFAR10'], ax=axes[0], palette=[sns.color_palette()[0], sns.color_palette()[3]])
    axes[0].set_xticklabels(['DET', 'VDP++'])
    # axes[0].set_yscale('log')
    sns.barplot(x='Model Type', y='Infidelity', data=explainability_df, ax=axes[1], order=['ModelType.DETCIFAR10', 'ModelType.VDPCIFAR10'], palette=[sns.color_palette()[0], sns.color_palette()[3]])
    axes[1].set_xticklabels(['DET', 'VDP++'])
    axes[1].set_yscale('log')
    plt.savefig('results/cifar10/cifar10_explainability.png')
    plt.clf()



if __name__ == '__main__':
    # eval_test_set()
    # snr_vs_gaussian_noise()
    # statistical_analysis()
    
    # plot_snr()
    # plot_prune()
    # plot_explainability()

    # prune_models()
    # explainability_analysis()
    test_acc_vs_gaussian_noise()