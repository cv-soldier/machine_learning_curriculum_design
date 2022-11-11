"""
Author: lyc
Created on 2022.10.30 13:00
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import time

train_loader = torch.utils.data.DataLoader(
                datasets.FashionMNIST("./data", train=True, download=True, transform=transforms.Compose([
                                                  transforms.ToTensor(),
                                                  transforms.Normalize((0.1307,), (0.3081,))
                                                  ])),
                batch_size=16, shuffle=True)
test_loader = torch.utils.data.DataLoader(
                datasets.FashionMNIST("./data", train=False, transform=transforms.Compose([
                                                  transforms.ToTensor(),
                                                  transforms.Normalize((0.1307,), (0.3081,))
                                                  ])),
                batch_size=16, shuffle=True)

class Model(nn.Module):
    def __init__(self, cuda):
        super(Model, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.bn1 = nn.BatchNorm2d(num_features=64)
        self.relu1 = nn.ReLU()
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(num_features=128)
        self.relu2 = nn.ReLU()
        self.fc5 = nn.Linear(in_features=128 * 8 * 8, out_features=512)
        self.drop1 = nn.Dropout()
        self.fc6 = nn.Linear(in_features=512, out_features=10)
        self.use_cuda = cuda
        self.cuda()

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.pool1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        x = self.conv3(x)
        x = self.conv4(x)
        x = self.pool2(x)
        x = self.bn2(x)
        x = self.relu2(x)

        x = x.view(-1, 128 * 8 * 8)
        x = F.relu(self.fc5(x))
        x = self.drop1(x)
        x = self.fc6(x)

        return F.log_softmax(x, dim=1)

def train(model, train_loader, epochs):
        model.train()
        optimizer = optim.SGD(model.parameters(), lr=0.005, momentum=0.9)
        for epoch in range(epochs):
            timestart = time.time()  # 希望全数据被训练几次
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.cuda(), target.cuda()
                optimizer.zero_grad()
                output = model(data)
                loss = F.nll_loss(output, target)
                loss.backward()
                optimizer.step()
                if batch_idx % 1000 == 0:
                    print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        epoch + 1, batch_idx * len(data), len(train_loader.dataset),
                        100. * batch_idx / len(train_loader), loss.item()))
            print('The epoch %d cost %3f sec' % (epoch + 1, time.time() - timestart))

def test(model, test_loader):
        model.eval()
        test_loss = 0
        correct = 0
        for data, target in test_loader:
            data, target = data.cuda(), target.cuda()
            output = model(data)
            test_loss += F.nll_loss(output, target).item()
            pred = output.data.max(1)[1] # get the index of the max log-probability
            correct += pred.eq(target.data).cpu().sum()

        test_loss = test_loss
        test_loss /= len(test_loader) # loss function already averages over batch size
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)))

if __name__=="__main__":
    cuda = True
    model = Model(cuda)
    train(model, train_loader, epochs=50)
    test(model, test_loader)
