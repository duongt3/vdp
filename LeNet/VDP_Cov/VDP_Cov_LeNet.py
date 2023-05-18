import time
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import torch.utils.data
from utils.EVI import EVI_Conv2D, EVI_FullyConnected, EVI_Relu, EVI_Softmax, EVI_Maxpool
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import roc_auc_score
from torch.nn.utils import prune
####################################################################################################
#
#   Author: Chris Angelini
#
#   Purpose: Extension of Dera et. Al. Bayesian eVI framework into Pytorch
#            The file is used for the creation of the eVI network structure and training loop
#
#   ToDo: Comment
#
####################################################################################################


class data_loader(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).long()

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        target = self.y[index]
        data_val = self.X[index, :]
        return data_val, target


class EVINet(nn.Module):
    def __init__(self, batch_size=512):
        super(EVINet, self).__init__()
        self.batch_size = batch_size
        self.conv1 = EVI_Conv2D(1, 6, padding=2, input_flag=True)
        self.conv2 = EVI_Conv2D(6, 16)
        self.conv3 = EVI_Conv2D(16, 120)
        self.fc1 = EVI_FullyConnected(120, 84)
        self.fc2 = EVI_FullyConnected(84, 10)
        self.pool = EVI_Maxpool(2, 2)
        self.relu = EVI_Relu()
        # self.bn1 = nn.BatchNorm1d(31)
        # self.bn2 = nn.BatchNorm1d(61)
        self.softmax = EVI_Softmax(1)

        self.register_buffer("thing", torch.tensor(1e-3).repeat([self.fc2.out_features]))

    def forward(self, x_input):
        # flat_x.requires_grad = True
        mu, sigma = self.conv1(x_input)
        mu, sigma = self.relu(mu, sigma)
        mu, sigma = self.pool(mu, sigma)

        mu, sigma = self.conv2(mu, sigma)
        mu, sigma = self.relu(mu, sigma)
        mu, sigma = self.pool(mu, sigma)

        mu, sigma = self.conv3(mu, sigma)
        mu, sigma = self.relu(mu, sigma)

        mu = torch.flatten(mu, 1)
        # sigma = torch.flatten(sigma, 1)

        mu, sigma = self.fc1(mu, sigma)
        mu, sigma = self.relu(mu, sigma)

        mu, sigma = self.fc2(mu, sigma)
        mu, sigma = self.softmax.forward(mu, sigma)

        return mu, sigma

    def nll_gaussian(self, y_pred_mean, y_pred_sd, y_test):
        thing = torch.tensor(1e-3)
        y_pred_sd_inv = torch.inverse(y_pred_sd + torch.diag(thing.repeat([self.fc2.out_features])).to(y_pred_sd.device))
        mu_ = y_pred_mean - torch.nn.functional.one_hot(y_test, num_classes=10)
        mu_sigma = torch.bmm(mu_.unsqueeze(1), y_pred_sd_inv)
        ms = 0.5 * torch.bmm(mu_sigma, mu_.unsqueeze(2)).squeeze(1) + 0.5 * torch.log(
            torch.det(y_pred_sd + torch.diag(thing.repeat([self.fc2.out_features])).to(y_pred_sd.device))).unsqueeze(1)
        ms = ms.mean()
        return ms

    def batch_loss(self, output_mean, output_sigma, label):
        output_sigma_clamp = torch.clamp(output_sigma, 1e-10, 1e+10)
        tau = 0.002
        log_likelihood = self.nll_gaussian(output_mean, output_sigma_clamp, label)
        loss_value = log_likelihood + tau * (self.conv1.kl_loss_term()+ self.conv2.kl_loss_term()+self.conv3.kl_loss_term()+self.fc1.kl_loss_term()+self.fc2.kl_loss_term())
        return loss_value

    def batch_accuracy(self, output_mean, label):
        _, bin = torch.max(output_mean.detach(), dim=1)
        comp = bin == label.detach()
        batch_accuracy = comp.sum().cpu().numpy()/len(label)
        return batch_accuracy

    # def score(self, X, y):
    #     predicted_labels = []
    #     testset = data_loader(X, y)
    #     testloader = DataLoader(testset, batch_size=self.batch_size, shuffle=False)
    #     for itr, (test_data, test_targets) in enumerate(testloader):
    #         test_data = test_data.float().to(torch.device('cuda:0'))
    #         y_pred, sig = self.forward(test_data)
    #         #predicted_batch = torch.argmax(y_pred, dim=1).cpu().numpy()
    #         predicted_labels.extend(y_pred.detach().cpu().numpy().tolist())

    #     score = roc_auc_score(y, np.asarray(predicted_labels)[:,1])
    #     return score

    def score(self, logits, y):
        score = torch.sum(torch.argmax(logits, dim=1) == y)/len(logits)
        return score.cpu().numpy()

    def train_model(self, epochs, trainloader, dir="/models/vdp_cov_model.pt"):

        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3, amsgrad=True)
        results = pd.DataFrame()

        for epoch in range(epochs):
            train_acc = list()
            model_size = torch.cuda.memory_allocated('cuda:0')
            start = time.time()
            for itr, (x, labels) in enumerate(trainloader):
                self.train()
                optimizer.zero_grad()
                mu, sigma = self.forward(x.float().to('cuda:0'))

                loss = self.batch_loss(mu, sigma, labels.to('cuda:0'))
                # log_det, nll = vdp.ELBOLoss(mu, sigma, labels.to('cuda:0'))
                # kl = vdp.gather_kl(self)
                # self.alpha, self.tau = vdp.scale_hyperp(log_det, nll, kl)
                # loss = self.alpha * log_det + nll + self.tau * 0.0001 * sum(kl)
                loss.backward()
                optimizer.step()

                print('Epoch {}/{}, itr {}: Training Loss: {:.2f}'.format(epoch+1, epochs, itr, loss))
                print('Train Accuracy: {:.2f}'.format(self.score(mu, labels.to('cuda:0'))))

            total_time = time.time()-start
            print('modelsize: {} trainingTime: {}'.format(model_size, total_time))
            results = results.append({'modelType': "VDP_Cov", 'epoch': epoch, 'modelSize': model_size, 'trainTime': total_time}, ignore_index=True)

            if epoch % 5 == 0:
                print('Saving Model...')
                torch.save(self.state_dict(), dir + "_epoch" + str(epoch) + ".pt")

        print('Saving Model...')
        torch.save(self.state_dict(), dir + ".pt")

        return results
    
    def prune_model(self, percentage, model_name):
        parameter = ((self.conv1.mean_conv, 'weight'),
                     (self.conv2.mean_conv, 'weight'),
                     (self.conv3.mean_conv, 'weight'),
                     (self.fc1.mean_fc, 'weight'),
                     (self.fc2.mean_fc, 'weight'))

        prune.global_unstructured(
            parameter,
            pruning_method=prune.L1Unstructured,
            amount=percentage)

        for module, param in parameter:
            prune.remove(module, param)

        torch.save(self.state_dict(), model_name)
