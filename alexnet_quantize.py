#!/home/sunhanbo/software/anaconda3/bin/python
#-*-coding:utf-8-*-
import torch
import torch.nn as nn
import numpy as np
import quantize

activation_bit = 0
weight_bit = 0
momentum = 0.707


class AlexNet(nn.Module):
    def __init__(self, num_classes):
        super(AlexNet, self).__init__()
        input_fix_config = {'input': {'mode': 'input'},
                            'weight': {'mode': 'weight', 'qbit': weight_bit},
                            'output': {'mode': 'activation_out', 'qbit': activation_bit, 'momentum': momentum},
                            }
        hidden_fix_config = {'input': {'mode': 'activation_in', 'qbit': activation_bit, 'momentum': momentum},
                             'weight': {'mode': 'weight', 'qbit': weight_bit},
                             'output': {'mode': 'activation_out', 'qbit': activation_bit, 'momentum': momentum},
                             }
        self.features = nn.Sequential(
            quantize.QuantizeConv2d(input_fix_config, 3, 64, kernel_size = 3, stride = 2, padding = 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace = True),
            nn.MaxPool2d(kernel_size = 2, stride = 2),
            quantize.QuantizeConv2d(hidden_fix_config, 64, 192, kernel_size = 3, stride = 1, padding = 1),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace = True),
            nn.MaxPool2d(kernel_size = 2, stride = 2),
            quantize.QuantizeConv2d(hidden_fix_config, 192, 384, kernel_size = 3, stride = 1, padding = 1),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace = True),
            quantize.QuantizeConv2d(hidden_fix_config, 384, 256, kernel_size = 3, stride = 1, padding = 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace = True),
            quantize.QuantizeConv2d(hidden_fix_config, 256, 256, kernel_size = 3, stride = 1, padding = 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace = True),
            nn.MaxPool2d(kernel_size = 2, stride = 2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 2 * 2, 4096),
            nn.ReLU(inplace = True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace = True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 256 * 2 * 2)
        x = self.classifier(x)
        return x

def get_net():
    net = AlexNet(10)
    return net

if __name__ == '__main__':
    net = get_net()
    print(net)
    for param in net.parameters():
        print(type(param.data), param.size())
    print('这是AlexNet网络，要求输入尺寸必为3x32x32，输出为10维分类结果')
