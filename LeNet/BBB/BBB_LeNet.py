import torch
import time
import pandas as pd
import numpy as np
from torch.nn import functional as F
from torch.nn.utils import prune

from . import BBBConv, BBBLinear, metrics, utils
from .misc import FlattenLayer, ModuleWrapper

class Model(ModuleWrapper):
    def __init__(self, priors):
        super(Model, self).__init__()
        self.priors = priors
        self.conv1 = BBBConv.BBBConv2d(1, 6, 5, padding=2, priors=self.priors)
        self.relu = torch.nn.ReLU()
        self.pool1 = torch.nn.MaxPool2d(2, 2)
        self.conv2 = BBBConv.BBBConv2d(6, 16, 5, priors=self.priors)
        self.relu = torch.nn.ReLU()
        self.pool2 = torch.nn.MaxPool2d(2, 2)
        self.conv3 = BBBConv.BBBConv2d(16, 120, 5, priors=self.priors)
        self.relu = torch.nn.ReLU()
        self.flatten = FlattenLayer(120)
        self.fc1 = BBBLinear.BBBLinear(120, 84, priors=self.priors)
        self.relu = torch.nn.ReLU()
        self.lin_last = BBBLinear.BBBLinear(84, 10, priors=self.priors)
        self.softmax = torch.nn.Softmax(dim=1)
    
    def score(self, logits, y):
        score = torch.sum(torch.argmax(logits, dim=1) == y)/len(logits)
        return score.cpu().numpy()

    def prune_model(self, percentage, model_name):
        parameter = ((self.conv1, 'W_mu'),
                     (self.conv2, 'W_mu'),
                     (self.conv3, 'W_mu'),
                     (self.fc1, 'W_mu'),
                     (self.lin_last, 'W_mu'))

        prune.global_unstructured(
            parameter,
            pruning_method=prune.L1Unstructured,
            amount=percentage)

        for module, param in parameter:
            prune.remove(module, param)

        torch.save(self.state_dict(), model_name)


    def train_model(self, epochs, trainloader, dir="/models/bbb_model.pt"):

        criterion = metrics.ELBO(len(trainloader.dataset)).to('cuda:0')
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3, amsgrad=True)

        lr_sched = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, verbose=True)
        
        results = pd.DataFrame()
        train_accs = []
        test_accs = []
        for epoch in range(epochs):
            self.train()
            model_size = torch.cuda.memory_allocated('cuda:0')
            start = time.time()
            for itr, (x, labels) in enumerate(trainloader):
                total_loss = 0
                optimizer.zero_grad()
                net_out, kl = self.forward(x.float().to('cuda:0'))
                outputs = torch.nn.functional.log_softmax(net_out, dim=1)

                beta = 1/len(x)
                loss = criterion(outputs, labels.to('cuda:0'), kl, beta)
                loss.backward()
                optimizer.step()
                train_acc = self.score(net_out, labels.to('cuda:0'))

                print('Epoch {}/{}, itr {}: Training Loss: {:.2f}'.format(epoch+1, epochs, itr, loss))
                print('Train Accuracy: {:.2f}'.format(train_acc))

            total_time = time.time()-start
            print('modelsize: {} trainingTime: {}'.format(model_size, total_time))
            results = results.append({'modelType': "BBB", 'epoch': epoch, 'modelSize': model_size, 'trainTime': total_time}, ignore_index=True)

            if epoch % 5 == 0:
                print('Saving Model...')
                torch.save(self.state_dict(), dir + "_epoch" + str(epoch) + ".pt")

            print('Saving Model...')
            torch.save(self.state_dict(), dir + ".pt")

        return results