#!/home/sunhanbo/software/anaconda3/bin/python
#-*-coding:utf-8-*-
import quantize
import torch
import torch.nn as nn
import numpy as np

activation_bit = 4
weight_bit = 8
momentum = 0.707

input_fix_config = {'input': {'mode': 'input'},
                    'weight': {'mode': 'weight', 'qbit': weight_bit},
                    'output': {'mode': 'activation_out', 'qbit': activation_bit, 'momentum': momentum},
                    }
hidden_fix_config = {'input': {'mode': 'activation_in', 'qbit': activation_bit, 'momentum': momentum},
                     'weight': {'mode': 'weight', 'qbit': weight_bit},
                     'output': {'mode': 'activation_out', 'qbit': activation_bit, 'momentum': momentum},
                     }

class VGGA(nn.Module):
    def __init__(self, num_classes):
        super(VGGA, self).__init__()
        # L1
        self.conv1 = quantize.QuantizeConv2d(input_fix_config, 3, 128, 3, padding = 1)
        self.bn1 = nn.BatchNorm2d(128)
        self.relu1 = nn.ReLU()
        # L2
        self.conv2 = quantize.QuantizeConv2d(hidden_fix_config, 128, 128, 3, padding = 1)
        self.bn2 = nn.BatchNorm2d(128)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size = 2, stride = 2)
        # L3
        self.conv3_1 = quantize.QuantizeConv2d(hidden_fix_config, 128, 256, 3, padding = 1)
        self.bn3_1 = nn.BatchNorm2d(256)
        self.relu3_1 = nn.ReLU()
        self.conv3_2 = quantize.QuantizeConv2d(hidden_fix_config, 256, 256, 3, padding = 1)
        self.bn3_2 = nn.BatchNorm2d(256)
        self.relu3_2 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(kernel_size = 2, stride = 2)
        # L4
        self.conv4_1 = quantize.QuantizeConv2d(hidden_fix_config, 256, 512, 3, padding = 1)
        self.bn4_1 = nn.BatchNorm2d(512)
        self.relu4_1 = nn.ReLU()
        self.conv4_2 = quantize.QuantizeConv2d(hidden_fix_config, 512, 512, 3, padding = 1)
        self.bn4_2 = nn.BatchNorm2d(512)
        self.relu4_2 = nn.ReLU()
        self.pool4 = nn.MaxPool2d(kernel_size = 2, stride = 2)
        # L5
        self.conv5 = quantize.QuantizeConv2d(hidden_fix_config, 512, 1024, 3)
        self.bn5 = nn.BatchNorm2d(1024)
        self.relu5 = nn.ReLU()
        self.pool5 = nn.MaxPool2d(kernel_size = 2, stride = 2)
        # fc
        self.fc = nn.Linear(1024, num_classes)

    def forward(self, x):
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.pool2(self.relu2(self.bn2(self.conv2(x))))
        x = self.relu3_1(self.bn3_1(self.conv3_1(x)))
        x = self.pool3(self.relu3_2(self.bn3_2(self.conv3_2(x))))
        x = self.relu4_1(self.bn4_1(self.conv4_1(x)))
        x = self.pool4(self.relu4_2(self.bn4_2(self.conv4_2(x))))
        x = self.pool5(self.relu5(self.bn5(self.conv5(x))))
        x = x.view(x.size(0), 1024)
        x = self.fc(x)
        return x

def get_net():
    net = VGGA(10)
    return net

if __name__ == '__main__':
    net = get_net()
    print(net)
    for param in net.parameters():
        print(type(param.data), param.size())
    print('这是VGGA网络，要求输入尺寸必为3x32x32，输出为10维分类结果')
