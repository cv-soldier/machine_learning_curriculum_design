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
        # 定义卷积层
        self.conv1 = torch.nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5)
        self.conv2 = torch.nn.Conv2d(in_channels=6, out_channels=12, kernel_size=5)
        # 定义全连接层
        self.fc1 = torch.nn.Linear(in_features=12*4*4, out_features=120)
        self.fc2 = torch.nn.Linear(in_features=120, out_features=60)
        self.fc3 = torch.nn.Linear(in_features=60, out_features=10)  # 10是固定的，因为必须要和模型所需要的分类个数一致
        self.use_cuda = cuda
        self.cuda()

    def forward(self, t):
        # 第一层卷积和池化处理
        t = self.conv1(t)
        t = F.relu(t)
        t = F.max_pool2d(t, kernel_size=2, stride=2)
        # 第二层卷积和池化处理
        t = self.conv2(t)
        t = F.relu(t)
        t = F.max_pool2d(t, kernel_size=2, stride=2)
        # 搭建全连接网络，第一层全连接
        t = t.reshape(-1, 12 * 4 * 4)  # 将卷积结果由4维变为2维
        t = self.fc1(t)
        t = F.relu(t)
        # 第二层全连接
        t = self.fc2(t)
        t = F.relu(t)
        # 第三层全连接
        t = self.fc3(t)
        return F.log_softmax(t, dim=1)

def train(model, train_loader, epochs):
        model.train()
        optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
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
