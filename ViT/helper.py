
import os
import vdp
import time
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
#from captum.attr import Saliency
from torchvision import transforms
from collections import defaultdict
from torch.utils.data import DataLoader
from scipy.stats import f_oneway, spearmanr
#from captum.metrics import infidelity, sensitivity_max
from torch.nn.utils import prune
from torchvision.datasets import ImageFolder
import pytorch_lightning as pl


from enum import Enum
import torchvision
import glob

def prune_det_parameters(model, percentage, save_directory):
    # These are the parameters for ResNet18, if pruning diff resnet need diff parameters
    parameter = ((model.model.encoder.layers.encoder_layer_0.ln_1, 'weight'),
                 (model.model.encoder.layers.encoder_layer_0.self_attention.out_proj, 'weight'),
                 (model.model.encoder.layers.encoder_layer_0.ln_2, 'weight'),
                 (model.model.encoder.layers.encoder_layer_0.mlp[0], 'weight'),
                 (model.model.encoder.layers.encoder_layer_0.mlp[3], 'weight'),
                 
                 (model.model.encoder.layers.encoder_layer_1.ln_1, 'weight'),
                 (model.model.encoder.layers.encoder_layer_1.self_attention.out_proj, 'weight'),
                 (model.model.encoder.layers.encoder_layer_1.ln_2, 'weight'),
                 (model.model.encoder.layers.encoder_layer_1.mlp[0], 'weight'),
                 (model.model.encoder.layers.encoder_layer_1.mlp[3], 'weight'),
                 
                 (model.model.encoder.layers.encoder_layer_2.ln_1, 'weight'),
                 (model.model.encoder.layers.encoder_layer_2.self_attention.out_proj, 'weight'),
                 (model.model.encoder.layers.encoder_layer_2.ln_2, 'weight'),
                 (model.model.encoder.layers.encoder_layer_2.mlp[0], 'weight'),
                 (model.model.encoder.layers.encoder_layer_2.mlp[3], 'weight'),
                 
                 (model.model.encoder.layers.encoder_layer_3.ln_1, 'weight'),
                 (model.model.encoder.layers.encoder_layer_3.self_attention.out_proj, 'weight'),
                 (model.model.encoder.layers.encoder_layer_3.ln_2, 'weight'),
                 (model.model.encoder.layers.encoder_layer_3.mlp[0], 'weight'),
                 (model.model.encoder.layers.encoder_layer_3.mlp[3], 'weight'),
                 
                 (model.model.encoder.layers.encoder_layer_4.ln_1, 'weight'),
                 (model.model.encoder.layers.encoder_layer_4.self_attention.out_proj, 'weight'),
                 (model.model.encoder.layers.encoder_layer_4.ln_2, 'weight'),
                 (model.model.encoder.layers.encoder_layer_4.mlp[0], 'weight'),
                 (model.model.encoder.layers.encoder_layer_4.mlp[3], 'weight'),
                 
                 (model.model.encoder.layers.encoder_layer_5.ln_1, 'weight'),
                 (model.model.encoder.layers.encoder_layer_5.self_attention.out_proj, 'weight'),
                 (model.model.encoder.layers.encoder_layer_5.ln_2, 'weight'),
                 (model.model.encoder.layers.encoder_layer_5.mlp[0], 'weight'),
                 (model.model.encoder.layers.encoder_layer_5.mlp[3], 'weight'),
                 
                 (model.model.encoder.layers.encoder_layer_6.ln_1, 'weight'),
                 (model.model.encoder.layers.encoder_layer_6.self_attention.out_proj, 'weight'),
                 (model.model.encoder.layers.encoder_layer_6.ln_2, 'weight'),
                 (model.model.encoder.layers.encoder_layer_6.mlp[0], 'weight'),
                 (model.model.encoder.layers.encoder_layer_6.mlp[3], 'weight'),
                 
                 (model.model.encoder.layers.encoder_layer_7.ln_1, 'weight'),
                 (model.model.encoder.layers.encoder_layer_7.self_attention.out_proj, 'weight'),
                 (model.model.encoder.layers.encoder_layer_7.ln_2, 'weight'),
                 (model.model.encoder.layers.encoder_layer_7.mlp[0], 'weight'),
                 (model.model.encoder.layers.encoder_layer_7.mlp[3], 'weight'),
                 
                 (model.model.encoder.layers.encoder_layer_8.ln_1, 'weight'),
                 (model.model.encoder.layers.encoder_layer_8.self_attention.out_proj, 'weight'),
                 (model.model.encoder.layers.encoder_layer_8.ln_2, 'weight'),
                 (model.model.encoder.layers.encoder_layer_8.mlp[0], 'weight'),
                 (model.model.encoder.layers.encoder_layer_8.mlp[3], 'weight'),
                 
                 (model.model.encoder.layers.encoder_layer_9.ln_1, 'weight'),
                 (model.model.encoder.layers.encoder_layer_9.self_attention.out_proj, 'weight'),
                 (model.model.encoder.layers.encoder_layer_9.ln_2, 'weight'),
                 (model.model.encoder.layers.encoder_layer_9.mlp[0], 'weight'),
                 (model.model.encoder.layers.encoder_layer_9.mlp[3], 'weight'),
                 
                 (model.model.encoder.layers.encoder_layer_10.ln_1, 'weight'),
                 (model.model.encoder.layers.encoder_layer_10.self_attention.out_proj, 'weight'),
                 (model.model.encoder.layers.encoder_layer_10.ln_2, 'weight'),
                 (model.model.encoder.layers.encoder_layer_10.mlp[0], 'weight'),
                 (model.model.encoder.layers.encoder_layer_10.mlp[3], 'weight'),
                 
                 (model.model.encoder.layers.encoder_layer_11.ln_1, 'weight'),
                 (model.model.encoder.layers.encoder_layer_11.self_attention.out_proj, 'weight'),
                 (model.model.encoder.layers.encoder_layer_11.ln_2, 'weight'),
                 (model.model.encoder.layers.encoder_layer_11.mlp[0], 'weight'),
                 (model.model.encoder.layers.encoder_layer_11.mlp[3], 'weight'))
    
    prune.global_unstructured(
        parameter,
        pruning_method=prune.L1Unstructured,
        amount=percentage)

    for module, param in parameter:
        prune.remove(module, param)

    torch.save(model.state_dict(), save_directory)

    return model

def prune_vdp_parameters(model, percentage, save_directory):
    # These are the parameters for ResNet18, if pruning diff resnet need diff parameters
    parameter = ((model.transformer.layers[0][0].norm.mu, 'weight'),
                    (model.transformer.layers[0][0].to_qkv.mu, 'weight'),
                    (model.transformer.layers[0][0].to_out.mu, 'weight'),
                    (model.transformer.layers[0][1].ln.mu, 'weight'),
                    (model.transformer.layers[0][1].lin1.mu, 'weight'),
                    (model.transformer.layers[0][1].lin2.mu, 'weight'),

                    (model.transformer.layers[1][0].norm.mu, 'weight'),
                    (model.transformer.layers[1][0].to_qkv.mu, 'weight'),
                    (model.transformer.layers[1][0].to_out.mu, 'weight'),
                    (model.transformer.layers[1][1].ln.mu, 'weight'),
                    (model.transformer.layers[1][1].lin1.mu, 'weight'),
                    (model.transformer.layers[1][1].lin2.mu, 'weight'),

                    (model.transformer.layers[2][0].norm.mu, 'weight'),
                    (model.transformer.layers[2][0].to_qkv.mu, 'weight'),
                    (model.transformer.layers[2][0].to_out.mu, 'weight'),
                    (model.transformer.layers[2][1].ln.mu, 'weight'),
                    (model.transformer.layers[2][1].lin1.mu, 'weight'),
                    (model.transformer.layers[2][1].lin2.mu, 'weight'),

                    (model.transformer.layers[3][0].norm.mu, 'weight'),
                    (model.transformer.layers[3][0].to_qkv.mu, 'weight'),
                    (model.transformer.layers[3][0].to_out.mu, 'weight'),
                    (model.transformer.layers[3][1].ln.mu, 'weight'),
                    (model.transformer.layers[3][1].lin1.mu, 'weight'),
                    (model.transformer.layers[3][1].lin2.mu, 'weight'),

                    (model.transformer.layers[4][0].norm.mu, 'weight'),
                    (model.transformer.layers[4][0].to_qkv.mu, 'weight'),
                    (model.transformer.layers[4][0].to_out.mu, 'weight'),
                    (model.transformer.layers[4][1].ln.mu, 'weight'),
                    (model.transformer.layers[4][1].lin1.mu, 'weight'),
                    (model.transformer.layers[4][1].lin2.mu, 'weight'),

                    (model.transformer.layers[5][0].norm.mu, 'weight'),
                    (model.transformer.layers[5][0].to_qkv.mu, 'weight'),
                    (model.transformer.layers[5][0].to_out.mu, 'weight'),
                    (model.transformer.layers[5][1].ln.mu, 'weight'),
                    (model.transformer.layers[5][1].lin1.mu, 'weight'),
                    (model.transformer.layers[5][1].lin2.mu, 'weight'),

                    (model.transformer.layers[6][0].norm.mu, 'weight'),
                    (model.transformer.layers[6][0].to_qkv.mu, 'weight'),
                    (model.transformer.layers[6][0].to_out.mu, 'weight'),
                    (model.transformer.layers[6][1].ln.mu, 'weight'),
                    (model.transformer.layers[6][1].lin1.mu, 'weight'),
                    (model.transformer.layers[6][1].lin2.mu, 'weight'),

                    (model.transformer.layers[7][0].norm.mu, 'weight'),
                    (model.transformer.layers[7][0].to_qkv.mu, 'weight'),
                    (model.transformer.layers[7][0].to_out.mu, 'weight'),
                    (model.transformer.layers[7][1].ln.mu, 'weight'),
                    (model.transformer.layers[7][1].lin1.mu, 'weight'),
                    (model.transformer.layers[7][1].lin2.mu, 'weight'),

                    (model.transformer.layers[8][0].norm.mu, 'weight'),
                    (model.transformer.layers[8][0].to_qkv.mu, 'weight'),
                    (model.transformer.layers[8][0].to_out.mu, 'weight'),
                    (model.transformer.layers[8][1].ln.mu, 'weight'),
                    (model.transformer.layers[8][1].lin1.mu, 'weight'),
                    (model.transformer.layers[8][1].lin2.mu, 'weight'),

                    (model.transformer.layers[9][0].norm.mu, 'weight'),
                    (model.transformer.layers[9][0].to_qkv.mu, 'weight'),
                    (model.transformer.layers[9][0].to_out.mu, 'weight'),
                    (model.transformer.layers[9][1].ln.mu, 'weight'),
                    (model.transformer.layers[9][1].lin1.mu, 'weight'),
                    (model.transformer.layers[9][1].lin2.mu, 'weight'),

                    (model.transformer.layers[10][0].norm.mu, 'weight'),
                    (model.transformer.layers[10][0].to_qkv.mu, 'weight'),
                    (model.transformer.layers[10][0].to_out.mu, 'weight'),
                    (model.transformer.layers[10][1].ln.mu, 'weight'),
                    (model.transformer.layers[10][1].lin1.mu, 'weight'),
                    (model.transformer.layers[10][1].lin2.mu, 'weight'),

                    (model.transformer.layers[11][0].norm.mu, 'weight'),
                    (model.transformer.layers[11][0].to_qkv.mu, 'weight'),
                    (model.transformer.layers[11][0].to_out.mu, 'weight'),
                    (model.transformer.layers[11][1].ln.mu, 'weight'),
                    (model.transformer.layers[11][1].lin1.mu, 'weight'),
                    (model.transformer.layers[11][1].lin2.mu, 'weight'),
                    )
    
    prune.global_unstructured(
        parameter,
        pruning_method=prune.L1Unstructured,
        amount=percentage)

    for module, param in parameter:
        prune.remove(module, param)

    torch.save(model.state_dict(), save_directory)

    return model