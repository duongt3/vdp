
import numpy as np
import torchvision
from torchvision.datasets import FashionMNIST, MNIST
from torchvision import *

from torch.utils.data import DataLoader
import pandas as pd

from captum.attr import Saliency
from captum.metrics import infidelity,  sensitivity_max

import argparse

from LeNet.BBB import BBB_LeNet
from LeNet.DET import DET_LeNet
from LeNet.VDP import VDP_LeNet
from LeNet.VDP_Cov import VDP_Cov_LeNet

from LeNet.BBB import config_bayesian

import matplotlib.pyplot as plt

import seaborn as sns
import scipy
from collections import defaultdict

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

def def_value():
    return 10000

def parseArgs():
    parser = argparse.ArgumentParser(description='Train models')
    parser.add_argument(action='store', dest='models', default='DET', help='List of models to train E.g. VDP,DET,BBB')
    parser.add_argument('-e', '--epochs', action='store', dest='epochs', type=int, default=10, help='Number of epochs to train the model')
    parser.add_argument('-bs', '--batch_size', action='store', dest='batch', type=int, default=128, help='Batch size to train the models')
    parser.add_argument('-d', '--data', action='store', dest='dataset', default='MNIST', help='Dataset to Train Default=MNIST E.g. MNIST FMNIST')

    return parser.parse_args()


def get_data_loaders(dataset, batch_sizeIn=4096):

    if "MNIST" == dataset:
        transform = transforms.Compose([transforms.ToTensor()]) #, transforms.Normalize((0.1307,), (0.3081,))])
        train = MNIST(os.getcwd(), train=True, download=True, transform=transform)
        test = MNIST(os.getcwd(), train=False, download=True, transform=transform)
        trainloader = DataLoader(train, batch_size=batch_sizeIn, num_workers=2,
                                 shuffle=True,
                                 pin_memory=True)  # IF YOU CAN FIT THE DATA INTO MEMORY DO NOT USE DATALOADERS
        testloader = DataLoader(test, batch_size=batch_sizeIn, num_workers=2, shuffle=True, pin_memory=True)
        input_channel = 1

    elif "FMNIST" == dataset:
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        train = FashionMNIST(os.getcwd(), train=True, download=True, transform=transform)
        test = FashionMNIST(os.getcwd(), train=False, download=True, transform=transform)
        trainloader = DataLoader(train, batch_size=batch_sizeIn, num_workers=2,
                                 shuffle=True,
                                 pin_memory=True)  # IF YOU CAN FIT THE DATA INTO MEMORY DO NOT USE DATALOADERS
        testloader = DataLoader(test, batch_size=batch_sizeIn, num_workers=2, shuffle=True, pin_memory=True)
        input_channel = 1

    elif "CIFAR10" == dataset:
        transform_test = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        train = torchvision.datasets.CIFAR10(os.getcwd(), train=True, download=True, transform=transform_train)
        test = torchvision.datasets.CIFAR10(os.getcwd(), train=False, download=True, transform=transform_test)
        trainloader = DataLoader(train, batch_size=batch_sizeIn, num_workers=2,
                                 shuffle=True,
                                 pin_memory=True)  # IF YOU CAN FIT THE DATA INTO MEMORY DO NOT USE DATALOADERS
        testloader = DataLoader(test, batch_size=batch_sizeIn, num_workers=2, shuffle=True, pin_memory=True)
        input_channel = 3

    # else:
        # Throw an exception

    return trainloader, testloader, input_channel


def get_model(model_type, input_channel):
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


def compute_sensitivity_max(model, test_loader):

    ig = Saliency(model)

    score = list()
    for itr, (x, labels) in enumerate(test_loader):
        attribution = ig.attribute(x.float().to('cuda:0'), target=labels.to('cuda:0'))
        score.append(sensitivity_max(ig.attribute, x.float().to('cuda:0'), perturb_radius=0.02, target=labels.to('cuda:0'), n_perturb_samples=5))

    return torch.mean(torch.cat(score, 0)).cpu().numpy().tolist()

def compute_infidelity_score(model, test_loader):
    # ImageClassifier takes a single input tensor of images Nx3x32x32,
    # and returns an Nx10 tensor of class probabilities.
    saliency = Saliency(model)
    score = list()

    for itr, (x, labels) in enumerate(test_loader):
        # Computes saliency maps for class 3.
        attribution = saliency.attribute(x.float().to('cuda:0'), target=labels.to('cuda:0'))
        # define a perturbation function for the input
        def perturb_fn(inputs):
            noise = torch.tensor(np.random.uniform(-0.03, 0.03, inputs.shape)).float().to('cuda:0')

            return noise, inputs - noise
        # Computes infidelity score for saliency maps
        score.append(infidelity(model, perturb_fn, x.float().to('cuda:0'), attribution, target=labels.to('cuda:0'), n_perturb_samples=5))

    return torch.mean(torch.cat(score, 0)).detach().cpu().numpy().tolist()

def plot_snr():
    train_loader, test_loader, input_channel = get_data_loaders('MNIST')
    model_list = ['VDP']
    for model_type in model_list:
        model = get_model(model_type, 1)

        def add_noise(s, snr):
            var_s = np.var(s, axis=1)
            var_n = var_s / (10 ** (snr / 10))
            rand_arr = np.random.randn(s.shape[0], s.shape[1])
            n = np.sqrt(var_n).reshape((-1, 1)) * rand_arr
            return s + n

        snrs = [-6, -3, 1, 5, 10, 20]
        sigmas = list()
        for snr in range(len(snrs)):
            inner_sigmas = list()
            for itr, (image, target) in enumerate(test_loader):
                batchsz = image.shape[0]
                image = image.reshape(batchsz, -1)

                image = add_noise(image.cpu().numpy(), snrs[snr]).reshape(batchsz, 1, 28, 28)
                mu, sigma = model.forward(torch.from_numpy(image).float().to('cuda:0'))
                inner_sigmas.append(torch.mean(sigma, dim=1).detach().cpu().numpy())
            inner_sigmas = np.hstack(inner_sigmas)
            sigmas.append(np.mean(inner_sigmas))
        plt.figure()
        plt.plot(snrs, sigmas)
        plt.xlabel('SNR (dB)')
        plt.ylabel('Mean Mean Test Sigma')
        plt.title(str(model_type))
        plt.show()
        plt.savefig(str(model_type)+'_snr.png')
        pass


def score(model, test):
    scores = 0
    for batch in test:
        x_test, y_test = batch
        logits = model(x_test.to('cuda:0'))
        if type(logits) == tuple:
            logits = logits[0]
        logits = torch.nn.functional.softmax(logits)
        scores += torch.sum(torch.argmax(logits, dim=1) == y_test.to('cuda:0'))

    return (scores / 10000).cpu().numpy()

def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return m, h

def main():

    # ARGUMENTS = parseArgs()

    # Configuration options
    num_runs = 5 # Number of times to run the experiment
    num_epochs = 50
    dataset = "MNIST"
    model_list = ['VDP']
    skip_training = False

    prune_percentages = [0, 0.25, 0.5, 0.9, 0.95, 0.96, 0.97, 0.98, 0.99]

    batch_sizes = defaultdict(def_value)
    batch_sizes['VDP_Cov'] = 512

    result = pd.DataFrame()
    for model_type in model_list:
        train_loader, test_loader, input_channel = get_data_loaders(dataset, batch_sizeIn=512)
        print(str(model_type))
        for run in range(num_runs):

            model_dir = 'models/mnist_models/' + model_type + '_run' + str(run)
            model = get_model(model_type, input_channel)

            if not skip_training:
                result = pd.concat([result, model.train_model(num_epochs, train_loader, dir=model_dir)], ignore_index=True)

            prune_model_dir = 'pruneModels/' + model_type + 'Prune_run' + str(run)
            if not os.path.exists(prune_model_dir):
                os.makedirs(prune_model_dir)

            # for percent in prune_percentages:
            #     prune_model_name = prune_model_dir + '/' + model_type + '_pruned_percent_' + str(percent) + '.pt'

            # For saliency computation, need to remove sigma from the return statement in the forward
            # if model_type == 'VDP':
            #     model = VDP_LeNet.VDPModel_Saliency(input_channel)
            #     model.to('cuda:0')
            # elif model_type == 'VDP_RES':
            #     model = ResNet18_Saliency()
            #     model.to('cuda:0')

            # model.load_state_dict(torch.load(model_dir))
            # model.prune_model(percent, prune_model_name)
            # test_accuracy = score(model, test_loader)
            # print('Test accuracy:' + str(test_accuracy))

            # sensitivity_score = compute_sensitivity_max(model, test_loader)
            # print('Sensitivity score: ' + str(sensitivity_score))

            # infid_score = compute_infidelity_score(model, test_loader)
            # print('Infidelity score: ' + str(infid_score))

            # result = result.append({'itr': run, 'Type': model_type, 'Sparsity Percentage': percent, 'Sensitivity Score': sensitivity_score, 'Infidelity Score': infid_score, 'Test Accuracy': test_accuracy}, ignore_index=True) #'test_accuracy': test_accuracy,

    result_file = 'results_' + dataset + '.csv'
    result.to_csv(result_file)

if __name__ =='__main__':
    # plot_snr()
    # get_KDE_by_Layers()
    main()

    # results = pd.read_csv('results_MNIST.csv')

    # prune_percentages = [0, 0.25, 0.5, 0.9, 0.95, 0.96, 0.97, 0.98]

    # VDP_upper, VDP_lower = list(), list()
    # DET_upper, DET_lower = list(), list()
    # BBB_upper, BBB_lower = list(), list()
    # for percentage in prune_percentages:
    #     m, h = mean_confidence_interval(results.loc[(results['Type'] == 'VDP') & (results['Sparsity Percentage'] == percentage)]['Test Accuracy'])
    #     VDP_upper.append(m+h)
    #     VDP_lower.append(m-h)

    #     m, h = mean_confidence_interval(results.loc[(results['Type'] == 'DET') & (results['Sparsity Percentage'] == percentage)]['Test Accuracy'])
    #     DET_upper.append(m+h)
    #     DET_lower.append(m-h)

    #     m, h = mean_confidence_interval(results.loc[(results['Type'] == 'BBB') & (results['Sparsity Percentage'] == percentage)]['Test Accuracy'])
    #     BBB_upper.append(m+h)
    #     BBB_lower.append(m-h)

    # plt.clf()
    # # plt.plot(prune_percentages, VDP_average_scores, label='VDP', color='blue')
    # # plt.plot(prune_percentages, DET_average_scores, label='Deterministic', color='black')

    # # plt.plot(prune_percentages, BBB_average_scores, label='BBB', color='red')
    # plt.fill_between(prune_percentages, VDP_upper, VDP_lower, color='tab:blue', alpha=0.5)
    # plt.fill_between(prune_percentages, DET_upper, DET_lower, color='tab:orange', alpha=0.5)
    # plt.fill_between(prune_percentages, BBB_upper, BBB_lower, color='tab:green', alpha=0.5)
    # plt.xlabel('Sparsity Percentage')
    # plt.ylabel('Test Accuracy')
    # plt.title('MNIST Small Networks Test Accuracy vs Global Pruning')
    # plt.legend()
    # plt.show()
    # plt.savefig('test_accuracy.png')
    # plt.clf()

    # plt.clf()
    # sns.barplot(x='Sparsity Percentage', y='Sensitivity Score', hue='Type', data=results)
    # plt.yscale('log')
    # plt.show()
    # plt.savefig('max_sensitivity_barplot.png')

    # plt.clf()
    # sns.barplot(x='Sparsity Percentage', y='Infidelity Score', hue='Type', data=results)
    # plt.yscale('log')
    # plt.show()
    # plt.savefig('infid_barplot.png')


