#!/home/sunhanbo/software/anaconda3/bin/python
#-*-coding:utf-8-*-
import torch
import torch.nn as nn
import numpy as np
import quantize

activation_bit = 0
weight_bit = 0
momentum = 0.707


class LeNet(nn.Module):
    def __init__(self, num_classes):
        super(LeNet, self).__init__()
        input_fix_config = {'input': {'mode': 'input'},
                            'weight': {'mode': 'weight', 'qbit': weight_bit},
                            'output': {'mode': 'activation_out', 'qbit': activation_bit, 'momentum': momentum},
                            }
        hidden_fix_config = {'input': {'mode': 'activation_in', 'qbit': activation_bit, 'momentum': momentum},
                             'weight': {'mode': 'weight', 'qbit': weight_bit},
                             'output': {'mode': 'activation_out', 'qbit': activation_bit, 'momentum': momentum},
                             }
        # L1
        self.c1 = quantize.QuantizePowerConv2d(input_fix_config, 3, 6, 5)
        self.b1 = nn.BatchNorm2d(6)
        self.r1 = nn.ReLU()
        self.s1 = nn.MaxPool2d(kernel_size = 2, stride = 2)
        # L2
        self.c2 = quantize.QuantizePowerConv2d(hidden_fix_config, 6, 16, 5)
        self.b2 = nn.BatchNorm2d(16)
        self.r2 = nn.ReLU()
        self.s2 = nn.MaxPool2d(kernel_size = 2, stride = 2)
        # L3
        self.c3 = quantize.QuantizePowerConv2d(hidden_fix_config, 16, 120, 5)
        self.b3 = nn.BatchNorm2d(120)
        self.r3 = nn.ReLU()
        # fc
        self.f4 = nn.Linear(120, 84)
        self.r4 = nn.ReLU()
        self.f5 = nn.Linear(84, num_classes)

    def forward(self, x):
        power = torch.zeros(1, dtype = torch.float, device = x.device)
        x = self.s1(self.r1(self.b1(self.c1(x, power))))
        x = self.s2(self.r2(self.b2(self.c2(x, power))))
        x = self.r3(self.b3(self.c3(x, power)))
        x = x.view(x.size(0), 120)
        x = self.f5(self.r4(self.f4(x)))
        return x, power

def get_net():
    net = LeNet(10)
    return net

if __name__ == '__main__':
    net = get_net()
    print(net)
    for param in net.parameters():
        print(type(param.data), param.size())
    print('这是LeNet网络，要求输入尺寸必为3x32x32，输出为10维分类结果')
