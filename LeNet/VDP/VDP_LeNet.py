import os
import sys
sys.path.append(os.getcwd())
import utils.vdp as vdp
import torch
import numpy as np
from torch.nn.utils import prune
import time
import pandas as pd

class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = vdp.Conv2d(1, 6, 5, padding=2, input_flag=True)
        self.conv2 = vdp.Conv2d(6, 16, 5)
        self.conv3 = vdp.Conv2d(16, 120, 5)
        self.fc1 = vdp.Linear(120, 84)  # 5*5 from image dimension
        self.pool = vdp.MaxPool2d(2, 2)
        self.relu = vdp.ReLU()
        self.lin_last = vdp.Linear(84, 10)
        self.softmax = vdp.Softmax()


    def forward(self, x):
        mu, sigma = self.conv1(x)
        mu, sigma = self.relu(mu, sigma)
        mu, sigma = self.pool(mu, sigma)

        mu, sigma = self.conv2(mu, sigma)
        mu, sigma = self.relu(mu, sigma)
        mu, sigma = self.pool(mu, sigma)

        mu, sigma = self.conv3(mu, sigma)
        mu, sigma = self.relu(mu, sigma)

        mu = torch.flatten(mu, 1)
        sigma = torch.flatten(sigma, 1)

        mu, sigma = self.fc1(mu, sigma)
        mu, sigma = self.relu(mu, sigma)

        mu, sigma = self.lin_last(mu, sigma)
        mu, sigma = self.softmax(mu, sigma)
        return mu, sigma


    def get_loss(self, mu, sigma, y):
        log_det, nll = vdp.ELBOLoss(mu, sigma, y)
        kl = vdp.gather_kl(self)
        # if self.alpha is None:
        #     self.alpha, self.tau = vdp.scale_hyperp(log_det, nll, kl)
        # loss = self.alpha * log_det + nll + self.tau * sum(kl)
        loss = log_det + 100 * nll + sum(kl)
        return loss

    def score(self, logits, y):
        score = torch.sum(torch.argmax(logits, dim=1) == y)/len(logits)
        return score.cpu().numpy()

    def prune_model(self, percentage, model_name):
        parameter = ((self.conv1.mu, 'weight'),
                     (self.conv2.mu, 'weight'),
                     (self.conv3.mu, 'weight'),
                     (self.fc1.mu, 'weight'),
                     (self.lin_last.mu, 'weight'))

        prune.global_unstructured(
            parameter,
            pruning_method=prune.L1Unstructured,
            amount=percentage)

        for module, param in parameter:
            prune.remove(module, param)

        torch.save(self.state_dict(), model_name)


    def train_model(self, epochs, trainloader, dir="/models/vdp_model.pt"):

        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3, amsgrad=True)
        results = pd.DataFrame()
        counter = 0
        max_training_acc = 0
        for epoch in range(epochs):
            train_acc = list()
            model_size = torch.cuda.memory_allocated('cuda:0')
            start = time.time()
            for itr, (x, labels) in enumerate(trainloader):
                self.train()
                optimizer.zero_grad()
                mu, sigma = self.forward(x.float().to('cuda:0'))

                loss = self.get_loss(mu, sigma, labels.to('cuda:0'))
                loss.backward()
                optimizer.step()

                training_acc = self.score(mu, labels.to('cuda:0'))
                

                print('Epoch {}/{}, itr {}: Training Loss: {:.2f}'.format(epoch+1, epochs, itr, loss))
                print('Train Accuracy: {:.2f}'.format(training_acc))
            
            if max_training_acc < training_acc:
                max_training_acc = max(max_training_acc, training_acc)
            elif (training_acc < max_training_acc) and (counter < 2):
                counter += 1
            elif (training_acc < max_training_acc) and (counter == 2):
                for g in optimizer.param_groups:
                    g['lr'] *= 0.5
                    print('Reduced LR by half')
                    counter = 0

            total_time = time.time()-start
            if epoch % 5 == 0:
                print('Saving Model...')
                torch.save(self.state_dict(), dir + "_epoch" + str(epoch) + ".pt")

        # print('Saving Model...')
        # torch.save(self.state_dict(), dir + ".pt")

        return results

class VDPModel_Saliency(torch.nn.Module):
    def __init__(self, input_channel=1):
        super(VDPModel_Saliency, self).__init__()
        self.alpha = 0.01
        self.tau = 0.001
        self.conv1 = vdp.Conv2d(1, 6, 5, padding=2, input_flag=True)
        self.conv2 = vdp.Conv2d(6, 16, 5)
        self.conv3 = vdp.Conv2d(16, 120, 5)
        self.fc1 = vdp.Linear(120, 84)  # 5*5 from image dimension
        self.pool = vdp.MaxPool2d(2, 2)
        self.relu = vdp.ReLU()
        self.lin_last = vdp.Linear(84, 10)
        self.softmax = vdp.Softmax()

        self.scale = False
        self.alpha, self.tau = None, None

        self.base_dir = "/models/vdp_model.pt"

    def forward(self, x):
        mu, sigma = self.conv1(x)
        mu, sigma = self.relu(mu, sigma)
        mu, sigma = self.pool(mu, sigma)

        mu, sigma = self.conv2(mu, sigma)
        mu, sigma = self.relu(mu, sigma)
        mu, sigma = self.pool(mu, sigma)

        mu, sigma = self.conv3(mu, sigma)
        mu, sigma = self.relu(mu, sigma)

        mu = torch.flatten(mu, 1)
        sigma = torch.flatten(sigma, 1)

        mu, sigma = self.fc1(mu, sigma)
        mu, sigma = self.relu(mu, sigma)

        mu, sigma = self.lin_last(mu, sigma)
        mu, sigma = self.softmax(mu, sigma)

        return mu

    def prune_model(self, percentage, model_name):
        parameter = ((self.conv1.mu, 'weight'),
                     (self.conv2.mu, 'weight'),
                     (self.conv3.mu, 'weight'),
                     (self.fc1.mu, 'weight'),
                     (self.lin_last.mu, 'weight'))

        prune.global_unstructured(
            parameter,
            pruning_method=prune.L1Unstructured,
            amount=percentage)

        for module, param in parameter:
            prune.remove(module, param)

        torch.save(self.state_dict(), model_name)