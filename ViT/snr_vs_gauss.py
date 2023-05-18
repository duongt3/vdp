import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
from torchvision import transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from simple_vit_vdp import SimpleViT_vdp


def add_noise(s, snr):
    var_s = np.var(s, axis=1)
    var_n = var_s / (10 ** (snr / 10))
    rand_arr = np.random.randn(s.shape[0], s.shape[1])
    n = np.sqrt(var_n).reshape((-1, 1)) * rand_arr
    return s + n
        
        
def snr_vs_gaussian_noise():
    snrs = [-6, -3, 1, 5, 10, 20, 40]
    transform = transforms.Compose([transforms.ToTensor()]) #, transforms.Normalize((0.1307,), (0.3081,))])
    test = MNIST('/data', train=False, download=False, transform=transform)
    results = list()
    testloader = DataLoader(test, batch_size=512, num_workers=2, shuffle=True, pin_memory=True)
    kf = 5
    model = SimpleViT_vdp(
    image_size = 28,
    patch_size = 7,
    channels = 1,
    num_classes = 10,
    dim = 64,
    depth = 6,
    heads = 8,
    mlp_dim = 128
    )
    model.load_state_dict(torch.load(f'mnist_vdp_vit.pt'))
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
            sigma = sigma.detach().cpu().numpy()
            uncertain = [sig[pred] for (pred, sig) in zip(preds, sigma)]
            inner_sigmas.append(uncertain)
        inner_sigmas = np.hstack(inner_sigmas)
        results.append({'SNR (dB)': snr, 'Mean Sigma': np.mean(inner_sigmas)})
        # pd.DataFrame(results).to_csv('experimental_results/mnist_snr.csv', index=False)
    snr_df = pd.DataFrame(results)
    plt.figure(figsize=(18, 10))
    ax = sns.lineplot(x='SNR (dB)', y='Mean Sigma', data=snr_df, legend=False, marker='o', linewidth=3, markersize=10)
    plt.ylabel('Mean Sigma')
    plt.savefig('mnist_sigma_vit.png')
    plt.clf()
    
    
if __name__ == '__main__':
    snr_vs_gaussian_noise()