#!/home/sunhanbo/software/anaconda3/bin/python
#-*-coding:utf-8-*-
import torch
import torch.nn as nn
import numpy as np

class ResNet18(nn.Module):
    def __init__(self, num_classes):
        super(ResNet18, self).__init__()
        # L1
        self.conv1 = nn.Conv2d(3, 16, 3, padding = 1)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu1 = nn.ReLU()
        # L2a
        self.conv2a_1 = nn.Conv2d(16, 16, 3, padding = 1)
        self.bn2a_1 = nn.BatchNorm2d(16)
        self.relu2a_1 = nn.ReLU()
        self.conv2a_2 = nn.Conv2d(16, 16, 3, padding = 1)
        self.bn2a_2 = nn.BatchNorm2d(16)
        self.relu2a = nn.ReLU()
        # L2b
        self.conv2b_1 = nn.Conv2d(16, 16, 3, padding = 1)
        self.bn2b_1 = nn.BatchNorm2d(16)
        self.relu2b_1 = nn.ReLU()
        self.conv2b_2 = nn.Conv2d(16, 16, 3, padding = 1)
        self.bn2b_2 = nn.BatchNorm2d(16)
        self.relu2b = nn.ReLU()
        # L3a
        self.conv3a_1 = nn.Conv2d(16, 16, 3, stride = 2, padding = 1)
        self.bn3a_1 = nn.BatchNorm2d(16)
        self.relu3a_1 = nn.ReLU()
        self.conv3a_2 = nn.Conv2d(16, 16, 3, padding = 1)
        self.bn3a_2 = nn.BatchNorm2d(16)
        self.conv3a_3 = nn.Conv2d(16, 16, 1, stride = 2)
        self.bn3a_3 = nn.BatchNorm2d(16)
        self.relu3a = nn.ReLU()
        # L3b
        self.conv3b_1 = nn.Conv2d(16, 16, 3, padding = 1)
        self.bn3b_1 = nn.BatchNorm2d(16)
        self.relu3b_1 = nn.ReLU()
        self.conv3b_2 = nn.Conv2d(16, 16, 3, padding = 1)
        self.bn3b_2 = nn.BatchNorm2d(16)
        self.relu3b = nn.ReLU()
        # L4a
        self.conv4a_1 = nn.Conv2d(16, 32, 3, stride = 2, padding = 1)
        self.bn4a_1 = nn.BatchNorm2d(32)
        self.relu4a_1 = nn.ReLU()
        self.conv4a_2 = nn.Conv2d(32, 32, 3, padding = 1)
        self.bn4a_2 = nn.BatchNorm2d(32)
        self.conv4a_3 = nn.Conv2d(16, 32, 1, stride = 2)
        self.bn4a_3 = nn.BatchNorm2d(32)
        self.relu4a = nn.ReLU()
        # L4b
        self.conv4b_1 = nn.Conv2d(32, 32, 3, padding = 1)
        self.bn4b_1 = nn.BatchNorm2d(32)
        self.relu4b_1 = nn.ReLU()
        self.conv4b_2 = nn.Conv2d(32, 32, 3, padding = 1)
        self.bn4b_2 = nn.BatchNorm2d(32)
        self.relu4b = nn.ReLU()
        # L5a
        self.conv5a_1 = nn.Conv2d(32, 64, 3, stride = 2, padding = 1)
        self.bn5a_1 = nn.BatchNorm2d(64)
        self.relu5a_1 = nn.ReLU()
        self.conv5a_2 = nn.Conv2d(64, 64, 3, padding = 1)
        self.bn5a_2 = nn.BatchNorm2d(64)
        self.conv5a_3 = nn.Conv2d(32, 64, 1, stride = 2)
        self.bn5a_3 = nn.BatchNorm2d(64)
        self.relu5a = nn.ReLU()
        # L5b
        self.conv5b_1 = nn.Conv2d(64, 64, 3, padding = 1)
        self.bn5b_1 = nn.BatchNorm2d(64)
        self.relu5b_1 = nn.ReLU()
        self.conv5b_2 = nn.Conv2d(64, 64, 3, padding = 1)
        self.bn5b_2 = nn.BatchNorm2d(64)
        self.relu5b = nn.ReLU()
        # avg pool
        self.avg_pool = nn.AvgPool2d(kernel_size = 4, stride = 4)
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x):
        # L1
        x = self.relu1(self.bn1(self.conv1(x)))
        # L2a
        x = self.relu2a(self.bn2a_2(self.conv2a_2(self.relu2a_1(self.bn2a_1(self.conv2a_1(x))))) + x)
        # L2b
        x = self.relu2b(self.bn2b_2(self.conv2b_2(self.relu2b_1(self.bn2b_1(self.conv2b_1(x))))) + x)
        # L3a
        x = self.relu3a(self.bn3a_2(self.conv3a_2(self.relu3a_1(self.bn3a_1(self.conv3a_1(x))))) + self.bn3a_3(self.conv3a_3(x)))
        # L3b
        x = self.relu3b(self.bn3b_2(self.conv3b_2(self.relu3b_1(self.bn3b_1(self.conv3b_1(x))))) + x)
        # L4a
        x = self.relu4a(self.bn4a_2(self.conv4a_2(self.relu4a_1(self.bn4a_1(self.conv4a_1(x))))) + self.bn4a_3(self.conv4a_3(x)))
        # L4b
        x = self.relu4b(self.bn4b_2(self.conv4b_2(self.relu4b_1(self.bn4b_1(self.conv4b_1(x))))) + x)
        # L5a
        x = self.relu5a(self.bn5a_2(self.conv5a_2(self.relu5a_1(self.bn5a_1(self.conv5a_1(x))))) + self.bn5a_3(self.conv5a_3(x)))
        # L5b
        x = self.relu5b(self.bn5b_2(self.conv5b_2(self.relu5b_1(self.bn5b_1(self.conv5b_1(x))))) + x)
        # avg pool
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

def get_net():
    net = ResNet18(10)
    return net

if __name__ == '__main__':
    net = get_net()
    print(net)
    for param in net.parameters():
        print(type(param.data), param.size())
    print('这是ResNet18网络，要求输入尺寸必为3x32x32，输出为10维分类结果')
