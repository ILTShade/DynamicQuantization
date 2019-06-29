#!/home/sunhanbo/software/anaconda3/bin/python
#-*-coding:utf-8-*-
import pdb
import torch
import torch.nn as nn
from torch.autograd import Function
import torch.nn.functional as F
import math

# METHOD = 'TRADITION'
# METHOD = 'FIX_TRAIN'
# METHOD = 'SPLIT_FIX_TRAIN'
METHOD = ''

class PowerConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, \
                 stride = 1, padding = 0, dilation = 1, groups = 1, bias = True):
        super(PowerConv2d, self).__init__()
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
    def forward(self, x):
        norm_x = x / torch.mean(torch.abs(x))
        norm_w = self.conv2d.weight / torch.mean(torch.abs(self.conv2d.weight))
        square_sum = F.conv2d(torch.mul(norm_x, norm_x), \
                              torch.abs(norm_w), \
                              None, \
                              self.conv2d.stride, \
                              self.conv2d.padding, \
                              self.conv2d.dilation, \
                              self.conv2d.groups, \
                              )
        return self.conv2d(x), torch.sum(square_sum)

# l = PowerConv2d(1, 1, 2)
# l.conv2d.weight.data.fill_(1)
# l.conv2d.bias.data.fill_(0)
# x = torch.arange(9.).reshape((1,1,3,3)).requires_grad_()
# y = l(x)
# print(y)

