#!/home/sunhanbo/software/anaconda3/bin/python
#-*-coding:utf-8-*-
import quantize
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class LeNet(nn.Module):
    def __init__(self, num_classes, weight_spec_bit_list, activation_spec_bit_list):
        super(LeNet, self).__init__()
        self.c1 = quantize.QuantizeConv2d(3, 6, 5, stride = 1, padding = 0,
                                          weight_spec_bit = weight_spec_bit_list[0],
                                          activation_spec_bit = activation_spec_bit_list[0],
                                          )
        self.relu1 = nn.ReLU()
        self.s1 = nn.MaxPool2d(kernel_size = 2, stride = 2)
        self.c2 = quantize.QuantizeConv2d(6, 16, 5, stride = 1, padding = 0,
                                          weight_spec_bit = weight_spec_bit_list[1],
                                          activation_spec_bit = activation_spec_bit_list[1],
                                          )
        self.relu2 = nn.ReLU()
        self.s2 = nn.MaxPool2d(kernel_size = 2, stride = 2)
        self.c3 = quantize.QuantizeConv2d(16, 120, 5, stride = 1, padding = 0,
                                          weight_spec_bit = weight_spec_bit_list[2],
                                          activation_spec_bit = activation_spec_bit_list[2],
                                          )
        self.relu3 = nn.ReLU()
        self.fc4 = nn.Linear(120, 84)
        self.relu4 = nn.ReLU()
        self.fc5 = nn.Linear(84, num_classes)

    def forward(self, x):
        quantize.last_activation_scale = 1 / 255
        quantize.last_activation_bit = 9

        x = self.s1(self.relu1(self.c1(x)))
        x = self.s2(self.relu2(self.c2(x)))
        x = self.relu3(self.c3(x))
        x = x.view(x.size(0), 120)
        x = self.relu4(self.fc4(x))
        x = self.fc5(x)
        return x

def get_net(weight_spec_bit_list = [9]*3, activation_spec_bit_list = [9]*3, config_name = None):
    print(f'采用输入参数对网络的参数进行配置，当前的训练模式为{quantize.METHOD}')
    print(weight_spec_bit_list)
    print(activation_spec_bit_list)
    net = LeNet(10, weight_spec_bit_list, activation_spec_bit_list)
    return net

if __name__ == '__main__':
    net = get_net()
    print(net)
    for param in net.parameters():
        print(type(param.data), param.size())
    print('这是LeNet网络，要求输入尺寸必为3x32x32，输出为10维分类结果')
    print(f'定点方式为{quantize.METHOD}')
