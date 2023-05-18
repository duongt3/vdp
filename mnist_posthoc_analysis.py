import os
import utils.vdp as vdp
import utils.EVI as EVI
import time
import torch
import numpy as np
import pandas as pd
from LeNet.VDP_Cov import VDP_Cov_LeNet
from tqdm import tqdm
import seaborn as sns
from LeNet.BBB import BBB_LeNet
from LeNet.DET import DET_LeNet
from LeNet.VDP import VDP_LeNet
import matplotlib.pyplot as plt
from captum.attr import Saliency, IntegratedGradients, NoiseTunnel, GuidedBackprop, InputXGradient
from torchvision import transforms
from collections import defaultdict
from LeNet.BBB import config_bayesian
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from scipy.stats import f_oneway, spearmanr
from captum.metrics import infidelity,  sensitivity_max

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="5"

def default_value():
    return 512


def get_model(model_type):
    if model_type == "VDP":
        model = VDP_LeNet.Model()
        model.to('cuda:0')

    elif model_type == "DET":
        model = DET_LeNet.Model()
        model.to('cuda:0')

    elif model_type == "BBB":
        model = BBB_LeNet.Model(priors = config_bayesian.priors)
        model.to('cuda:0')

    elif model_type == "VDP_Cov":
        model = VDP_Cov_LeNet.EVINet()
        model.to('cuda:0')
        
    return model


def eval_test_set(models_to_evaluate, path_to_model='models/mnist_models'):
    metrics = list()
    kf = 5  # LOAD KF DIFF MODELS
    batch_sizes = defaultdict(default_value)
    batch_sizes['VDP_Cov'] = 32
    for model_type in tqdm(models_to_evaluate):
        model = get_model(model_type)
        transform = transforms.Compose([transforms.ToTensor()]) #, transforms.Normalize((0.1307,), (0.3081,))])
        test = MNIST(os.getcwd(), train=False, download=True, transform=transform)
        testloader = DataLoader(test, batch_size=batch_sizes[model_type], num_workers=2, shuffle=True, pin_memory=True)
        for k in tqdm(range(kf), leave=False):
            start_time = time.time()
            model.load_state_dict(torch.load(f'{path_to_model}/{model_type}_run{k}.pt'))
            model.eval()
            model.to('cuda:0')
            logits, labels = list(), list()
            for itr, (x, y) in enumerate(testloader):
                x = x.to('cuda:0')
                if model_type == 'DET':
                    logits.append(model(x).detach().cpu())
                elif model_type == 'BBB':
                    net_out, _ = model.forward(x.float())
                    logits.append(torch.nn.functional.log_softmax(net_out, dim=1).detach().cpu())
                elif model_type == 'VDP':
                    mu, _ = model(x)
                    logits.append(mu.detach().cpu())
                elif model_type == 'VDP_Cov':
                    mu, _ = model(x)
                    logits.append(mu.detach().cpu())
                labels.append(y)
            # print(model.score(torch.cat(logits), torch.cat(labels)))
            test_accuracy = model.score(torch.cat(logits), torch.cat(labels))
            metrics.append({'Model Type': model_type, 'KF': k, 'Test Accuracy': test_accuracy, 'Inference Time': time.time() - start_time})
        pd.DataFrame(metrics).to_csv('experimental_results/mnist_test_results.csv', index=False)
    pd.DataFrame(metrics).to_csv('experimental_results/mnist_test_results.csv', index=False)


def add_noise(s, snr):
    var_s = np.var(s, axis=1)
    var_n = var_s / (10 ** (snr / 10))
    rand_arr = np.random.randn(s.shape[0], s.shape[1])
    n = np.sqrt(var_n).reshape((-1, 1)) * rand_arr
    return s + n
        
        
def snr_vs_gaussian_noise():
    model_list = ['VDP_Cov', 'VDP']
    snrs = [-6, -3, 1, 5, 10, 20, 40]
    batch_sizes = defaultdict(default_value)
    batch_sizes['VDP_Cov'] = 64
    transform = transforms.Compose([transforms.ToTensor()]) #, transforms.Normalize((0.1307,), (0.3081,))])
    test = MNIST(os.getcwd(), train=False, download=True, transform=transform)
    results = list()
    for model_type in tqdm(model_list):
        model = get_model(model_type)
        testloader = DataLoader(test, batch_size=batch_sizes[model_type], num_workers=2, shuffle=True, pin_memory=True)
        kf = 5
        for k in tqdm(range(kf), leave=False):
            model.load_state_dict(torch.load(f'models/{model_type}_run{k}.pt'))
            model.eval()
            model.to('cuda:0')
            for snr in tqdm(snrs, leave=False):
                inner_sigmas = list()
                for _, (image, _) in enumerate(testloader):
                    cur_batch_size = image.shape[0]
                    image = image.reshape(cur_batch_size, -1)
                    image = add_noise(image.numpy(), snr).reshape(cur_batch_size, 1, 28, 28)
                    mu, sigma = model.forward(torch.from_numpy(image).float().to('cuda:0'))
                    preds = torch.argmax(mu, dim=1).detach().cpu().numpy()
                    if model_type == 'VDP':
                        sigma = sigma.detach().cpu().numpy()
                    elif model_type == 'VDP_Cov':
                        sigma = torch.diagonal(sigma, dim1=1, dim2=2).detach().cpu().numpy()
                    uncertain = [sig[pred] for (pred, sig) in zip(preds, sigma)]
                    inner_sigmas.append(uncertain)
                inner_sigmas = np.hstack(inner_sigmas)
                results.append({'Model Type': model_type, 'KF':k, 'SNR (dB)': snr, 'Mean Sigma': np.mean(inner_sigmas)})
                pd.DataFrame(results).to_csv('experimental_results/mnist_snr.csv', index=False)
                
                
def test_acc_vs_gaussian_noise():
    model_list = ['DET', 'BBB', 'VDP_Cov', 'VDP']
    snrs = [-6, -3, 1, 5, 10, 20, 40]
    batch_sizes = defaultdict(default_value)
    batch_sizes['VDP_Cov'] = 64
    transform = transforms.Compose([transforms.ToTensor()]) #, transforms.Normalize((0.1307,), (0.3081,))])
    test = MNIST(os.getcwd(), train=False, download=True, transform=transform)
    results = list()
    for model_type in tqdm(model_list):
        model = get_model(model_type)
        testloader = DataLoader(test, batch_size=batch_sizes[model_type], num_workers=2, shuffle=True, pin_memory=True)
        kf = 5
        for k in tqdm(range(kf), leave=False):
            model.load_state_dict(torch.load(f'models/mnist_models/{model_type}_run{k}.pt'))
            model.eval()
            model.to('cuda:0')
            for snr in tqdm(snrs, leave=False):
                predicted = list()
                truth = list()
                for _, (image, labels) in enumerate(testloader):
                    cur_batch_size = image.shape[0]
                    image = image.reshape(cur_batch_size, -1)
                    image = add_noise(image.numpy(), snr).reshape(cur_batch_size, 1, 28, 28)
                    if model_type in ['VDP_Cov', 'VDP', 'BBB']:
                        mu, _ = model.forward(torch.from_numpy(image).float().to('cuda:0'))
                        preds = torch.argmax(mu, dim=1).detach().cpu().numpy().tolist()
                    else:
                        preds = model.forward(torch.from_numpy(image).float().to('cuda:0'))
                        preds = torch.argmax(preds, dim=1).detach().cpu().numpy().tolist()
                    predicted.append(preds)
                    truth.append(labels.numpy().tolist())
                predicted = np.hstack(predicted)
                truth = np.hstack(truth)
                test_acc = sum([p == t for (p, t) in zip(predicted, truth)]) / len(truth)
                results.append({'Model Type': model_type, 'KF':k, 'SNR (dB)': snr, 'Test Acc': test_acc})
                pd.DataFrame(results).to_csv('results/mnist/experimental_results/mnist_snr_test_acc.csv', index=False)
    

def get_layers_vdp(model):
    layers = list()
    for name, layer in model.named_children():
        if type(layer) == vdp.Conv2d or type(layer) == vdp.Linear:
            mus = layer.mu.weight.reshape(-1, 1).detach().cpu().numpy().squeeze()
            sigmas = layer.sigma.weight.reshape(-1, 1).detach().cpu().numpy().squeeze()
            [layers.append({'Layer Name': name, 'Mu': mu}) for mu in mus]
            [layers.append({'Layer Name': name, 'Sigma': sigma}) for sigma in sigmas]
    return layers


def get_layers_vdp_cov(model):
    layers = list()
    for name, layer in model.named_children():
        if type(layer) == VDP_Cov.EVI_Conv2D:
            mus = layer.mean_conv.weight.reshape(-1, 1).detach().cpu().numpy().squeeze()
            sigmas = layer.sigma_conv_weight.reshape(-1, 1).detach().cpu().numpy().squeeze()
            [layers.append({'Layer Name': name, 'Mu': mu}) for mu in mus]
            [layers.append({'Layer Name': name, 'Sigma': sigma}) for sigma in sigmas]
        elif type(layer) == VDP_Cov.EVI_FullyConnected:
            mus = layer.mean_fc.weight.reshape(-1, 1).detach().cpu().numpy().squeeze()
            sigmas = layer.sigma_fc_weight.reshape(-1, 1).detach().cpu().numpy().squeeze()
            [layers.append({'Layer Name': name, 'Mu': mu}) for mu in mus]
            [layers.append({'Layer Name': name, 'Sigma': sigma}) for sigma in sigmas]

    return layers


def get_layers_det(model):
    layers = list()
    for name, layer in model.named_children():
        if type(layer) == torch.nn.modules.conv.Conv2d or type(layer) == torch.nn.modules.linear.Linear:
            weights = layer.weight.reshape(-1, 1).detach().cpu().numpy().squeeze()
            [layers.append({'Layer Name': name, 'Weight': weight}) for weight in weights]
    return layers


def compute_sensitivity_max(model, test_loader):

    ig = Saliency(model.to('cuda:0'))
    # ig = InputXGradient(model.to('cuda:0'))

    score = list()
    for itr, (x, labels) in enumerate(test_loader):
        nt = NoiseTunnel(ig)
        # attribution = ig.attribute(x.float().to('cuda:0'), target=labels.to('cuda:0'))
        # attribution = nt.attribute(x.float().to('cuda:0'), nt_type='smoothgrad', nt_samples=10, target=labels.to('cuda:0'))
        score.append(sensitivity_max(ig.attribute, x.float().to('cuda:0'), perturb_radius=0.1, target=labels.to('cuda:0'), n_perturb_samples=5).detach().cpu().numpy())
        # score.append(sensitivity_max(nt.attribute, x.float().to('cuda:0'), perturb_radius=0.1, target=labels.to('cuda:0'), n_perturb_samples=5).detach().cpu().numpy())

    return np.mean(np.concatenate(score))


def compute_infidelity_score(model, test_loader):
    # ImageClassifier takes a single input tensor of images Nx3x32x32,
    # and returns an Nx10 tensor of class probabilities.
    saliency = Saliency(model.to('cuda:0'))
    # saliency = InputXGradient(model.to('cuda:0'))
    score = list()

    for itr, (x, labels) in enumerate(test_loader):
        # nt = NoiseTunnel(saliency)
        # Computes saliency maps for class 3.
        attribution = saliency.attribute(x.float().to('cuda:0'), target=labels.to('cuda:0'))
        # attribution = nt.attribute(x.float().to('cuda:0'), nt_type='smoothgrad', nt_samples=10, target=labels.to('cuda:0'))
        # define a perturbation function for the input
        def perturb_fn(inputs):
            noise = torch.tensor(np.random.uniform(-0.5, 0.5, inputs.shape)).float().to('cuda:0')

            return noise, inputs - noise
        # Computes infidelity score for saliency maps
        score.append(infidelity(model, perturb_fn, x.float().to('cuda:0'), attribution, target=labels.to('cuda:0'), n_perturb_samples=5).detach().cpu().numpy())

    return np.mean(np.concatenate(score))


def weight_distribution_analysis():
    model_types = ['VDP']
    epochs = [0, 25, 45]
    palette = sns.color_palette('colorblind')
    for model_type in model_types:
        fig, axes = plt.subplots(5, len(epochs), figsize=(18, 10), sharey=False)
        for itr, epoch in enumerate(epochs):
            model = get_model(model_type)
            model.load_state_dict(torch.load(f'models/mnist_models/{model_type}_run0_epoch{epoch}.pt'))
            if model_type == 'DET':
                layers = get_layers_det(model)
                layers = pd.DataFrame(layers)
                for num, layer in enumerate(layers['Layer Name'].unique()):
                    to_plot = layers.loc[layers['Layer Name'] == layer]
                    sns.kdeplot(data=to_plot['Weight'].squeeze(), ax=axes[num, itr], shade=True, fill=True, color=palette[num])
                    axes[num, itr].legend([layer])
            elif model_type == 'VDP':
                layers = get_layers_vdp(model)
                layers = pd.DataFrame(layers)
                for num, layer in enumerate(layers['Layer Name'].unique()):
                    to_plot = layers.loc[layers['Layer Name'] == layer]
                    sns.kdeplot(data=to_plot['Mu'].squeeze(), ax=axes[num, itr], shade=True, fill=True, color=palette[num])
                    # sns.kdeplot(data=to_plot['Sigma'].squeeze(), ax=axes[num, itr], shade=True, fill=True)
                    axes[num, itr].legend([layer])
            elif model_type == 'VDP_Cov':
                layers = get_layers_vdp_cov(model)
                layers = pd.DataFrame(layers)
                for num, layer in enumerate(layers['Layer Name'].unique()):
                    to_plot = layers.loc[layers['Layer Name'] == layer]
                    sns.kdeplot(data=to_plot['Mu'].squeeze(), ax=axes[num, itr], shade=True, fill=True, color=palette[num])
                    axes[num, itr].legend([layer])
                # sns.kdeplot(x='Sigma', data=layers, hue='Layer Name', ax=axes[itr], shade=True, fill=True)
    axes[0, 0].set_title('Epoch 0')
    axes[0, 1].set_title('Epoch 25')
    axes[0, 2].set_title('Epoch 45')     
    plt.suptitle(f'{model_type}')       
    plt.show()
    

def explainability_analysis():
    models_to_evaluate = ['DET', 'VDP', 'BBB']
    kf = 5  # LOAD KF DIFF MODELS
    batch_sizes = defaultdict(default_value)
    batch_sizes['VDP_Cov'] = 64
    scores = list()
    for model_type in tqdm(models_to_evaluate):
        model = get_model(model_type)
        transform = transforms.Compose([transforms.ToTensor()]) #, transforms.Normalize((0.1307,), (0.3081,))])
        test = MNIST(os.getcwd(), train=False, download=True, transform=transform)
        testloader = DataLoader(test, batch_size=batch_sizes[model_type], num_workers=2, shuffle=True, pin_memory=True)
        for k in tqdm(range(kf), leave=False):
            model.load_state_dict(torch.load(f'models/mnist_models/{model_type}_run{k}.pt'))
            model.to('cuda:0')
            sensitivity = compute_sensitivity_max(model, testloader)
            infidelity = compute_infidelity_score(model, testloader)
            scores.append({'Model Type': model_type, 'KF':k, 'Sensitivity Max':sensitivity, 'Infidelity': infidelity})
            pd.DataFrame(scores).to_csv('results/mnist/experimental_results/mnist_explainability.csv', index=False)


def pruning_analysis(path_to_model='models/mnist_models'):
    model_list = ['DET', 'BBB', 'VDP_Cov', 'VDP']
    model_dir = 'pruned_models'
    prune_percentages = [0, 0.25, 0.5, 0.75, 0.9, 0.95, 0.96, 0.97, 0.98]
    kf = 5
    for model_type in model_list:
        model = get_model(model_type)
        for k in range(kf):
            model.load_state_dict(torch.load(f'{path_to_model}/{model_type}_run{k}.pt'))
            for percentage in prune_percentages:
                str_percent = str(int(percentage * 100))
                if not os.path.exists(f'{model_dir}/{model_type}_{k}_{str_percent}.pt'):
                    model.prune_model(percentage, f'{model_dir}/{model_type}_{k}_{str_percent}.pt')
    metrics = list()
    kf = 5  # LOAD KF DIFF MODELS
    batch_sizes = defaultdict(default_value)
    batch_sizes['VDP_Cov'] = 64
    for model_type in tqdm(model_list):
        model = get_model(model_type)
        transform = transforms.Compose([transforms.ToTensor()]) #, transforms.Normalize((0.1307,), (0.3081,))])
        test = MNIST(os.getcwd(), train=False, download=True, transform=transform)
        testloader = DataLoader(test, batch_size=batch_sizes[model_type], num_workers=2, shuffle=True, pin_memory=True)
        for k in tqdm(range(kf), leave=False):
            for percentage in tqdm(prune_percentages, leave=False):
                str_percent = str(int(percentage * 100))
                model.load_state_dict(torch.load(f'{model_dir}/{model_type}_{k}_{str_percent}.pt'))
                model.eval()
                model.to('cuda:0')
                logits, labels = list(), list()
                for itr, (x, y) in enumerate(testloader):
                    x = x.to('cuda:0')
                    if model_type == 'DET':
                        logits.append(model(x).detach().cpu())
                    elif model_type == 'BBB':
                        net_out, _ = model.forward(x.float())
                        logits.append(torch.nn.functional.log_softmax(net_out, dim=1).detach().cpu())
                    elif model_type == 'VDP':
                        mu, _ = model(x)
                        logits.append(mu.detach().cpu())
                    elif model_type == 'VDP_Cov':
                        mu, _ = model(x)
                        logits.append(mu.detach().cpu())
                    labels.append(y)
                test_accuracy = model.score(torch.cat(logits), torch.cat(labels))
                metrics.append({'Model Type': model_type, 'KF': k, 'Percentage of Parameters Pruned': str_percent, 'Test Accuracy': test_accuracy})
                pd.DataFrame(metrics).to_csv('experimental_results/mnist_pruning.csv', index=False)
    pass
    

def plot_test_things():
    # Accuracy Plot
    test_df = pd.read_csv('results/mnist/experimental_results/mnist_test_results.csv')
    plt.figure(figsize=(18, 10))
    sns.set(font_scale=2)
    test_df['Model Type'] = pd.Categorical(test_df['Model Type'])
    ax = sns.barplot(x='Model Type', y='Test Accuracy', data=test_df, order=['DET', 'BBB', 'VDP_Cov', 'VDP'])
    ax.set_xticklabels(['DET', 'BBB', 'VDP Cov', 'VDP++'])
    plt.savefig('results/mnist/experimental_results/mnist_test_accuracy.png')
    plt.clf()
    
    # GPU Util
    max_gpu_utilization = [
        {'Model Type': 'DET', 'Max GPU Memory Usage (GB)': 1.591},
        {'Model Type': 'BBB', 'Max GPU Memory Usage (GB)': 1.593},
        {'Model Type': 'VDP_Cov', 'Max GPU Memory Usage (GB)': 38.809},
        {'Model Type': 'VDP++', 'Max GPU Memory Usage (GB)': 1.901}
    ]
    plt.figure(figsize=(18, 10))
    ax = sns.barplot(x='Model Type', y='Max GPU Memory Usage (GB)', data=pd.DataFrame(max_gpu_utilization), order=['DET', 'BBB', 'VDP_Cov', 'VDP++'])
    ax.set_xticklabels(['DET', 'BBB', 'VDP Cov', 'VDP++'])
    plt.savefig('results/mnist/experimental_results/mnist_gpu_usage.png')
    plt.clf()
    
    # # Epoch time Figure is from 512 batch size no lr annealing
    # train_df = pd.read_csv('experimental_results/results_MNIST.csv')
    # plt.figure(figsize=(18, 10))
    # ax = sns.barplot(x='modelType', y='trainTime', data=train_df, order=['DET', 'BBB', 'VDP_Cov', 'VDP'])
    # ax.set_xticklabels(['DET', 'BBB', 'VDP Cov', 'VDP++'])
    # ax.set_xlabel('Model Type')
    # ax.set_ylabel('Average Epoch Time')
    # plt.savefig('experimental_results/mnist_epoch_time.png')
    # plt.clf()
    
    # SNR plot
    snr_df = pd.read_csv('results/mnist/experimental_results/mnist_snr.csv')
    plt.figure(figsize=(18, 10))
    # REMOVE -6
    # snr_df = snr_df.loc[snr_df['SNR (dB)'] != -6]
    for model_type in snr_df['Model Type'].unique():
        for k in snr_df['KF'].unique():
            filter = (snr_df['Model Type'] == model_type) & (snr_df['KF'] == k)
            snr_df.loc[filter, 'Mean Sigma'] = snr_df.loc[filter]['Mean Sigma'] / snr_df.loc[filter]['Mean Sigma'].iloc[0]
    ax = sns.lineplot(x='SNR (dB)', y='Mean Sigma', data=snr_df, hue='Model Type', legend=True, marker='o', linewidth=3, markersize=10, palette=[sns.color_palette()[2], sns.color_palette()[3]])
    # plt.legend(title='Model Type', labels=['VDP Cov', 'VDP++'])
    legend_labels, _= ax.get_legend_handles_labels()
    ax.legend(legend_labels, ['VDP Cov', 'VDP++'])
    plt.ylabel('Normalized Mean Sigma')
    plt.savefig('results/mnist/experimental_results/mnist_sigma.png')
    plt.clf()
    
    
    # Explainability Plots
    explainability_df = pd.read_csv('results/mnist/experimental_results/mnist_explainability.csv')
    fig, axes = plt.subplots(1, 2, figsize=(18, 10))
    sns.barplot(x='Model Type', y='Sensitivity Max', data=explainability_df, order=['DET', 'BBB', 'VDP'], ax=axes[0], palette=[sns.color_palette()[0], sns.color_palette()[1], sns.color_palette()[3]])
    axes[0].set_xticklabels(['DET', 'BBB', 'VDP++'])
    axes[0].set_yscale('log')
    sns.barplot(x='Model Type', y='Infidelity', data=explainability_df, order=['DET', 'BBB', 'VDP'], ax=axes[1], palette=[sns.color_palette()[0], sns.color_palette()[1], sns.color_palette()[3]])
    axes[1].set_xticklabels(['DET', 'BBB', 'VDP++'])
    axes[1].set_yscale('log')
    plt.savefig('results/mnist/experimental_results/mnist_explainability.png')
    plt.clf()
    
    # Pruning Plots
    pruning_df = pd.read_csv('results/mnist/experimental_results/mnist_pruning.csv')
    all_prune = list([
        {'Model Type': 'DET', 'KF': 0, 'Percentage of Parameters Pruned': 100, 'Test Accuracy': 0},
        {'Model Type': 'BBB', 'KF': 0, 'Percentage of Parameters Pruned': 100, 'Test Accuracy': 0}, 
        {'Model Type': 'VDP_Cov', 'KF': 0, 'Percentage of Parameters Pruned': 100, 'Test Accuracy': 0},
        {'Model Type': 'VDP', 'KF': 0, 'Percentage of Parameters Pruned': 100, 'Test Accuracy': 0},      
    ])
    pruning_df = pd.concat([pruning_df, pd.DataFrame(all_prune)], ignore_index=True)
    plt.figure(figsize=(18, 10))
    ax = sns.lineplot(x='Percentage of Parameters Pruned', y='Test Accuracy', data=pruning_df, hue='Model Type', legend=False, marker='o', linewidth=3, markersize=10)
    plt.legend(title='Model Type', labels=['DET', 'BBB', 'VDP Cov', 'VDP++'])
    plt.savefig('results/mnist/experimental_results/mnist_pruning.png')
    plt.clf()

    noise_df = pd.read_csv('results/mnist/experimental_results/mnist_snr_test_acc.csv')
    model_types = ['DET', 'BBB', 'VDP_Cov', 'VDP']
    for model_type in model_types:
        filt = noise_df.query('`Model Type` == @model_type and `SNR (dB)` == -3')
        m = filt['Test Acc'].mean()
        std = filt['Test Acc'].std()
        print(f'{model_type}: Test Acc: {m} +/- {std}')
    plt.figure(figsize=(18, 10))
    ax = sns.barplot(x='Model Type', y='Test Acc', data=noise_df.query('`SNR (dB)` == -6'))
    plt.ylabel('Test Accuracy @-6dB SNR')
    ax.set_xticklabels(['DET', 'BBB', 'VDP Cov', 'VDP++'])
    # legend_labels, _= ax.get_legend_handles_labels()
    # ax.legend(legend_labels, ['DET', 'BBB', 'VDP Cov', 'VDP++'])
    # plt.legend(title='Model Type', labels=['DET', 'BBB', 'VDP Cov', 'VDP++'])
    plt.savefig('results/mnist/experimental_results/mnist_snr_test_acc.png')
    plt.clf()
    

def test_performance_analysis():
    if not os.path.exists('results/mnist/experimental_results/mnist_test_results.csv'):
        eval_test_set(['VDP_Cov', 'BBB', 'VDP', 'DET'])
    if not os.path.exists('results/mnist/experimental_results/mnist_snr.csv'):
     snr_vs_gaussian_noise()
    if not os.path.exists('results/mnist/experimental_results/mnist_explainability.csv'):
        explainability_analysis()
    if not os.path.exists('results/mnist/experimental_results/mnist_pruning.csv'):
        pruning_analysis()
    plot_test_things()

    
def statistical_analysis():
    test_df = pd.read_csv('results/mnist/experimental_results/mnist_test_results.csv')
    vdp_cov = test_df.loc[test_df['Model Type']=='VDP_Cov', 'Test Accuracy'].tolist()
    vdp = test_df.loc[test_df['Model Type']=='VDP', 'Test Accuracy'].tolist()
    det = test_df.loc[test_df['Model Type']=='DET', 'Test Accuracy'].tolist()
    bbb = test_df.loc[test_df['Model Type']=='BBB', 'Test Accuracy'].tolist()
    print('Test statistical analysis')
    print(f_oneway(vdp_cov, vdp, det, bbb))
    snr_df = pd.read_csv('results/mnist/experimental_results/mnist_snr.csv')
    snr_df = snr_df.loc[snr_df['SNR (dB)'] != -6]
    # for model_type in snr_df['Model Type'].unique():
    #     for k in snr_df['KF'].unique():
    #         filter = (snr_df['Model Type'] == model_type) & (snr_df['KF'] == k)
    #         snr_df.loc[filter, 'Mean Sigma'] = snr_df.loc[filter]['Mean Sigma'] / snr_df.loc[filter]['Mean Sigma'].iloc[0]
    vdp_cov = snr_df.loc[snr_df['Model Type'] == 'VDP_Cov', 'Mean Sigma']
    vdp_cov = np.array([vdp_cov[i::5] for i in range(5)])
    vdp = snr_df.loc[snr_df['Model Type'] == 'VDP', 'Mean Sigma']
    vdp = np.array([vdp[i::5] for i in range(5)])
    print('SNR statistical analysis')
    print([spearmanr(vdp_cov[i], vdp[i]) for i in range(len(vdp))])
    pass


if __name__ == '__main__':
    # test_performance_analysis()
    # weight_distribution_analysis()
    statistical_analysis()
    # test_acc_vs_gaussian_noise()
    # plot_test_things()
    # plot_snr_test_acc()
    pass