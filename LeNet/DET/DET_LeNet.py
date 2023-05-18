import torch
import time
import pandas as pd
from torch.nn.utils import prune

class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 6, 5, padding=2)
        self.conv2 = torch.nn.Conv2d(6, 16, 5)
        self.conv3 = torch.nn.Conv2d(16, 120, 5)
        self.fc1 = torch.nn.Linear(120, 84)  # 5*5 from image dimension
        self.pool = torch.nn.MaxPool2d(2, 2)
        self.relu = torch.nn.ReLU()
        self.lin_last = torch.nn.Linear(84, 10)
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, x):
        device = 'cuda:0' if next(self.parameters()).is_cuda else 'cpu'
        if not torch.is_tensor(x):
            x = torch.tensor(x, requires_grad=True, device=device, dtype=torch.float32)
        if x.device.type != device:
            x = x.to(device)
        # Max pooling over a (2, 2) window
        x = self.pool(self.relu(self.conv1(x)))
        # If the size is a square, you can specify with a single number
        x = self.pool(self.relu(self.conv2(x)))
        x = self.relu(self.conv3(x))
        x = torch.flatten(x, 1)  # flatten all dimensions except the batch dimension
        x = self.relu(self.fc1(x))
        x = self.lin_last(x)
        return x

    def prune_model(self, percentage, model_name):
        parameter = ((self.conv1, 'weight'),
                     (self.conv2, 'weight'),
                     (self.conv3, 'weight'),
                     (self.fc1, 'weight'),
                     (self.lin_last, 'weight'))

        prune.global_unstructured(
            parameter,
            pruning_method=prune.L1Unstructured,
            amount=percentage)

        for module, param in parameter:
            prune.remove(module, param)

        torch.save(self.state_dict(), model_name)

    def train_model(self, epochs, trainloader, dir="/models/det_model.pt"):

        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3, amsgrad=True)
        criterion = torch.nn.CrossEntropyLoss()
        results = pd.DataFrame()
        for epoch in range(epochs):
            train_acc = list()
            model_size = torch.cuda.memory_allocated('cuda:0')
            start = time.time()
            for itr, (x, labels) in enumerate(trainloader):
                self.train()
                optimizer.zero_grad()
                logits = self.forward(x.float().to('cuda:0'))
                loss = criterion(logits, labels.to('cuda:0'))

                loss.backward()
                optimizer.step()

                print('Epoch {}/{}, itr {}: Training Loss: {:.2f}'.format(epoch+1, epochs, itr, loss))
                print('Train Accuracy: {:.2f}'.format(self.score(logits, labels.to('cuda:0'))))

            total_time = time.time()-start
            print('modelsize: {} trainingTime: {}'.format(model_size, total_time))
            results = results.append({'modelType': "DET", 'epoch': epoch, 'modelSize': model_size, 'trainTime': total_time}, ignore_index=True)

            if epoch % 5 == 0:
                print('Saving Model...')
                torch.save(self.state_dict(), dir + "_epoch" + str(epoch) + ".pt")

        print('Saving Model...')
        torch.save(self.state_dict(), dir + ".pt")

        return results

    def score(self, logits, y):
        score = torch.sum(torch.argmax(logits, dim=1) == y)/len(logits)
        return score.cpu().numpy()

