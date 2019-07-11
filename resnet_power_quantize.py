#!/home/sunhanbo/software/anaconda3/bin/python
#-*-coding:utf-8-*-
import quantize
import torch
import torch.nn as nn
import numpy as np

activation_bit = 0
weight_bit = 0
momentum = 0.707


class ResNet18(nn.Module):
    def __init__(self, num_classes):
        super(ResNet18, self).__init__()
        input_fix_config = {'input': {'mode': 'input'},
                            'weight': {'mode': 'weight', 'qbit': weight_bit},
                            'output': {'mode': 'activation_out', 'qbit': activation_bit, 'momentum': momentum},
                            }
        hidden_fix_config = {'input': {'mode': 'activation_in', 'qbit': activation_bit, 'momentum': momentum},
                             'weight': {'mode': 'weight', 'qbit': weight_bit},
                             'output': {'mode': 'activation_out', 'qbit': activation_bit, 'momentum': momentum},
                             }
        # L1
        self.conv1 = quantize.QuantizePowerConv2d(input_fix_config, 3, 16, 3, padding = 1)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu1 = nn.ReLU()
        # L2a
        self.conv2a_1 = quantize.QuantizePowerConv2d(hidden_fix_config, 16, 16, 3, padding = 1)
        self.bn2a_1 = nn.BatchNorm2d(16)
        self.relu2a_1 = nn.ReLU()
        self.conv2a_2 = quantize.QuantizePowerConv2d(hidden_fix_config, 16, 16, 3, padding = 1)
        self.bn2a_2 = nn.BatchNorm2d(16)
        self.relu2a = nn.ReLU()
        # L2b
        self.conv2b_1 = quantize.QuantizePowerConv2d(hidden_fix_config, 16, 16, 3, padding = 1)
        self.bn2b_1 = nn.BatchNorm2d(16)
        self.relu2b_1 = nn.ReLU()
        self.conv2b_2 = quantize.QuantizePowerConv2d(hidden_fix_config, 16, 16, 3, padding = 1)
        self.bn2b_2 = nn.BatchNorm2d(16)
        self.relu2b = nn.ReLU()
        # L3a
        self.conv3a_1 = quantize.QuantizePowerConv2d(hidden_fix_config, 16, 16, 3, stride = 2, padding = 1)
        self.bn3a_1 = nn.BatchNorm2d(16)
        self.relu3a_1 = nn.ReLU()
        self.conv3a_2 = quantize.QuantizePowerConv2d(hidden_fix_config, 16, 16, 3, padding = 1)
        self.bn3a_2 = nn.BatchNorm2d(16)
        self.conv3a_3 = quantize.QuantizePowerConv2d(hidden_fix_config, 16, 16, 1, stride = 2)
        self.bn3a_3 = nn.BatchNorm2d(16)
        self.relu3a = nn.ReLU()
        # L3b
        self.conv3b_1 = quantize.QuantizePowerConv2d(hidden_fix_config, 16, 16, 3, padding = 1)
        self.bn3b_1 = nn.BatchNorm2d(16)
        self.relu3b_1 = nn.ReLU()
        self.conv3b_2 = quantize.QuantizePowerConv2d(hidden_fix_config, 16, 16, 3, padding = 1)
        self.bn3b_2 = nn.BatchNorm2d(16)
        self.relu3b = nn.ReLU()
        # L4a
        self.conv4a_1 = quantize.QuantizePowerConv2d(hidden_fix_config, 16, 32, 3, stride = 2, padding = 1)
        self.bn4a_1 = nn.BatchNorm2d(32)
        self.relu4a_1 = nn.ReLU()
        self.conv4a_2 = quantize.QuantizePowerConv2d(hidden_fix_config, 32, 32, 3, padding = 1)
        self.bn4a_2 = nn.BatchNorm2d(32)
        self.conv4a_3 = quantize.QuantizePowerConv2d(hidden_fix_config, 16, 32, 1, stride = 2)
        self.bn4a_3 = nn.BatchNorm2d(32)
        self.relu4a = nn.ReLU()
        # L4b
        self.conv4b_1 = quantize.QuantizePowerConv2d(hidden_fix_config, 32, 32, 3, padding = 1)
        self.bn4b_1 = nn.BatchNorm2d(32)
        self.relu4b_1 = nn.ReLU()
        self.conv4b_2 = quantize.QuantizePowerConv2d(hidden_fix_config, 32, 32, 3, padding = 1)
        self.bn4b_2 = nn.BatchNorm2d(32)
        self.relu4b = nn.ReLU()
        # L5a
        self.conv5a_1 = quantize.QuantizePowerConv2d(hidden_fix_config, 32, 64, 3, stride = 2, padding = 1)
        self.bn5a_1 = nn.BatchNorm2d(64)
        self.relu5a_1 = nn.ReLU()
        self.conv5a_2 = quantize.QuantizePowerConv2d(hidden_fix_config, 64, 64, 3, padding = 1)
        self.bn5a_2 = nn.BatchNorm2d(64)
        self.conv5a_3 = quantize.QuantizePowerConv2d(hidden_fix_config, 32, 64, 1, stride = 2)
        self.bn5a_3 = nn.BatchNorm2d(64)
        self.relu5a = nn.ReLU()
        # L5b
        self.conv5b_1 = quantize.QuantizePowerConv2d(hidden_fix_config, 64, 64, 3, padding = 1)
        self.bn5b_1 = nn.BatchNorm2d(64)
        self.relu5b_1 = nn.ReLU()
        self.conv5b_2 = quantize.QuantizePowerConv2d(hidden_fix_config, 64, 64, 3, padding = 1)
        self.bn5b_2 = nn.BatchNorm2d(64)
        self.relu5b = nn.ReLU()
        # avg pool
        self.avg_pool = nn.AvgPool2d(kernel_size = 4, stride = 4)
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x):
        power = torch.zeros(1, dtype = torch.float, device = x.device)
        # L1
        x = self.relu1(self.bn1(self.conv1(x, power)))
        # L2a
        x1 = self.relu2a_1(self.bn2a_1(self.conv2a_1(x, power)))
        x2 = self.bn2a_2(self.conv2a_2(x1, power))
        x = self.relu2a(x2 + x)
        # L2b
        x1 = self.relu2b_1(self.bn2b_1(self.conv2b_1(x, power)))
        x2 = self.bn2b_2(self.conv2b_2(x1, power))
        x = self.relu2b(x2 + x)
        # L3a
        x1 = self.relu3a_1(self.bn3a_1(self.conv3a_1(x, power)))
        x2 = self.bn3a_2(self.conv3a_2(x1, power))
        x3 = self.bn3a_3(self.conv3a_3(x, power))
        x = self.relu3a(x2 + x3)
        # L3b
        x1 = self.relu3b_1(self.bn3b_1(self.conv3b_1(x, power)))
        x2 = self.bn3b_2(self.conv3b_2(x1, power))
        x = self.relu3b(x2 + x)
        # L4a
        x1 = self.relu4a_1(self.bn4a_1(self.conv4a_1(x, power)))
        x2 = self.bn4a_2(self.conv4a_2(x1, power))
        x3 = self.bn4a_3(self.conv4a_3(x, power))
        x = self.relu4a(x2 + x3)
        # L4b
        x1 = self.relu4b_1(self.bn4b_1(self.conv4b_1(x, power)))
        x2 = self.bn4b_2(self.conv4b_2(x1, power))
        x = self.relu4b(x2 + x)
        # L5a
        x1 = self.relu5a_1(self.bn5a_1(self.conv5a_1(x, power)))
        x2 = self.bn5a_2(self.conv5a_2(x1, power))
        x3 = self.bn5a_3(self.conv5a_3(x, power))
        x = self.relu5a(x2 + x3)
        # L5b
        x1 = self.relu5b_1(self.bn5b_1(self.conv5b_1(x, power)))
        x2 = self.bn5b_2(self.conv5b_2(x1, power))
        x = self.relu5b(x2 + x)
        # avg pool
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x, power

def get_net():
    net = ResNet18(10)
    return net

if __name__ == '__main__':
    net = get_net()
    print(net)
    for param in net.parameters():
        print(type(param.data), param.size())
    print('这是ResNet18网络，要求输入尺寸必为3x32x32，输出为10维分类结果')
