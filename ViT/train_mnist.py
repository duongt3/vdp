import torch
from tqdm import tqdm
from simple_vit import SimpleViT
from simple_vit_vdp import SimpleViT_vdp
from torchvision import transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader


def train_model():
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
    transform = transforms.Compose([transforms.ToTensor()]) #, transforms.Normalize((0.1307,), (0.3081,))])
    train = MNIST('/data', train=True, download=True, transform=transform)
    # test = MNIST(os.getcwd(), train=False, download=True, transform=transform)
    trainloader = DataLoader(train, batch_size=512, num_workers=2,
                                shuffle=True,
                                pin_memory=True)  # IF YOU CAN FIT THE DATA INTO MEMORY DO NOT USE DATALOADERS
    # testloader = DataLoader(test, batch_size=4096, num_workers=2, shuffle=True, pin_memory=True)
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
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, amsgrad=True)
    criterion = torch.nn.CrossEntropyLoss()
    no_epochs = 10
    model.to(device)
    train_acc = 0
    for _ in tqdm(range(no_epochs)):
        for _, (x, labels) in tqdm(enumerate(trainloader), desc=f'Training Accuracy: {train_acc:.2f}', leave=False):
            optimizer.zero_grad()
            mu, sigma = model(x.float().to(device))
            # preds = model(x.float().to(device))
            # loss = criterion(preds, labels.to(device))

            loss = model.get_loss(mu, sigma, labels.to(device))
            loss.backward()
            optimizer.step()

            # train_acc = model.score(preds, labels.to(device))         
            train_acc = model.score(mu, labels.to(device))         

    print(train_acc)
    # print('Saving Model...')
    torch.save(model.state_dict(), "mnist_vdp_vit.pt")
        
        
if __name__ == '__main__':
    train_model()