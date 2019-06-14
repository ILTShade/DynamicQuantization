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

MAX_BIT = 8
MAX_NUM = 256
count_one_tensor = torch.zeros(MAX_NUM)
for i in range(MAX_NUM):
    count = 0
    tmp = i
    while True:
        if tmp == 0:
            break
        tmp = tmp & (tmp - 1)
        count = count + 1
    count_one_tensor[i] = count
# 得到在BIT数限制下的对应位置
src_tgt_tensor = torch.zeros(MAX_NUM)
for i in range(MAX_NUM):
    if count_one_tensor[i] > MAX_BIT:
        for j in range(MAX_NUM):
            if i + j < MAX_NUM:
                if count_one_tensor[i + j] <= MAX_BIT:
                    target = i + j
                    break
            if i - j >= 0:
                if count_one_tensor[i - j] <= MAX_BIT:
                    target = i - j
                    break
        else:
            assert 0
    else:
        target = i
    src_tgt_tensor[i] = target

def CountOne(tensor):
    src = count_one_tensor.to(dtype = tensor.dtype, device = tensor.device, copy = True)
    indices = tensor.to(dtype = torch.int64, copy = True)
    assert torch.max(indices) < MAX_NUM
    return torch.take(src, indices)

def TransOne(tensor):
    src = count_one_tensor.to(dtype = tensor.dtype, device = tensor.device, copy = True)
    indices = tensor.to(dtype = torch.int64, copy = True)
    assert torch.max(indices) < MAX_NUM
    return torch.take(src, indices)

# last_activation_scale, last_weight_scale 分别代表activation和weight的放缩细数
# last_activation_bit, last_weight_bit 分别代表activation和weight的比特数
# 真实值和定点数值的关系是 真实值 / scale 是 [-2 ^ (q - 1) + 1, 2 ^ (q - 1) - 1]
global last_activation_scale
global last_weight_scale
global last_activation_bit
global last_weight_bit
# quantize function
class QuantizeFunction(Function):
    @staticmethod
    def forward(ctx, input, qbit, mode, last_value = None, training = None):
        # input 是输入
        # qbit 是定点的长度
        # mode 是定点的模式，只能出现'weight', 'activation'
        # last_value 是上次的最大值
        # training 是train的标志
        global last_weight_scale
        global last_activation_scale
        global last_weight_bit
        global last_activation_bit
        if mode == 'weight':
            last_weight_bit = qbit
            scale = torch.max(torch.abs(input)).item()
        elif mode == 'activation':
            last_activation_bit = qbit
            if training:
                ratio = 0.81
                tmp = last_value.item()
                tmp = ratio * tmp + (1 - ratio) * torch.max(torch.abs(input)).item()
                last_value.data[0] = tmp
            scale = last_value.data[0]
        else:
            assert 0
        # transfer
        thres = 2 ** (qbit - 1) - 1
        output = input / scale
        output = torch.clamp(torch.round(output * thres), 0 - thres, thres - 0)
        if mode == 'weight' and METHOD == 'SPLIT_FIX_TRAIN':
            src = src_tgt_tensor.to(dtype = output.dtype, device = output.device, copy = True)
            indices = output.to(dtype = torch.int64, copy = True)
            torch.max(indices) < MAX_NUM
            a = torch.take(src, indices)
            print(a - output)
        output = output * scale / thres
        if mode == 'weight':
            last_weight_scale = scale / thres
        elif mode == 'activation':
            last_activation_scale = scale / thres
        else:
            assert 0
        return output
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None, None, None, None
Quantize = QuantizeFunction.apply
# a = torch.tensor([0., 1., 2., 3.]).requires_grad_().cuda()
# b = Quantize(a, 4, 'weight')
# c = Quantize(b, 5, 'weight')
# s = c.sum()
# s.backward()

# single function
class SingleFunction(Function):
    @staticmethod
    def forward(ctx, input):
        output = torch.round(input)
        output = torch.clamp(output, -1, 1)
        return output
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output
Single = SingleFunction.apply

# 量化卷积层
class QuantizeConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride = 1, padding = 0, dilation = 1, groups = 1, bias = True,
                 weight_spec_bit = 9, activation_spec_bit = 9):
        super(QuantizeConv2d, self).__init__()
        # 对不同的METHOD有不同的初始化方法
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size,
                                stride = stride, padding = padding, dilation = dilation, groups = groups,
                                bias = True)
        self.batch_norm = nn.batch_norm = nn.BatchNorm2d(out_channels)
        if METHOD == 'FIX_TRAIN' or METHOD == 'SPLIT_FIX_TRAIN':
            self.last_value = nn.Parameter(torch.zeros(1))
        self.weight_spec_bit = weight_spec_bit
        self.activation_spec_bit = activation_spec_bit
        self.stride = stride
        self.padding  = padding
        self.dilation = dilation
        self.groups = groups

    def forward(self, input):
        # 传统方法，直接计算即可
        if METHOD == 'TRADITION':
            return self.batch_norm(self.conv2d(input))
        # 简单直接的定点方法
        elif METHOD == 'FIX_TRAIN' or METHOD == 'SPLIT_FIX_TRAIN':
            weight = Quantize(self.conv2d.weight,
                              self.weight_spec_bit,
                              'weight',
                              None,
                              None,
                              )
            output = F.conv2d(input,
                              weight,
                              self.conv2d.bias,
                              self.stride,
                              self.padding,
                              self.dilation,
                              self.groups,
                              )
            return self.batch_norm(Quantize(output,
                                            self.activation_spec_bit,
                                            'activation',
                                            self.last_value,
                                            self.training,
                                            )
                                   )
        else:
            assert 0
    def extra_repr(self):
        return f'weight_spec_bit: {self.weight_spec_bit}, activation_spec_bit: {self.activation_spec_bit}'

if __name__ == '__main__':
    pass
    # test QuantizeConv2d
    # TRADITION
    # l = QuantizeConv2d(1, 1, 2, bias = False)
    # l.conv2d.weight.data.fill_(1)
    # a = torch.arange(4.).reshape((1, 1, 2, 2)).requires_grad_()
    # s = torch.sum(l(a))
    # s.backward()
    # print(s)
    # print(a.grad)
    # FIX_TRAIN
    # l = QuantizeConv2d(1, 1, 2, bias = False, weight_spec_bit = 3, activation_spec_bit = 3)
    # l.conv2d.weight.data.fill_(1)
    # l.conv2d.weight.data[0, 0, 0, 0] = 2
    # a = torch.arange(4.).reshape((1, 1, 2, 2)).requires_grad_()
    # s = torch.sum(l(a))
    # s.backward()
    # print(s)
    # print(a.grad)
    # SPLIT_FIX_TRAIN
    # l = QuantizeConv2d(1, 1, 2, bias = False, weight_spec_bit = 3, activation_spec_bit = 3)
    # a = torch.arange(4.).reshape((1, 1, 2, 2)).requires_grad_()
    # s = torch.sum(l(a))
    # s.backward()
    # print(s)
    # print(l.scale.grad)
    # print(a.grad)

    # test QuantizeLinear
    # a = torch.arange(27.).reshape((3, 1, 3, 3)).requires_grad_()
    # b = a.view(a.shape[0], -1)
    # l = QuantizeLinear(9, 2)
    # l.linear_list[0].weight.data.fill_(1)
    # l.linear_list[1].weight.data.fill_(1)
    # l.linear_list[2].weight.data.fill_(1)
    # l.last_value.data[0] = 127.
    # l.eval()
    # print(l)
    # print(l(b))
    # s = torch.sum(l(b))
    # s.backward()
    # print(a.grad)

    # test QuantizeBias
    # a = torch.arange(16.).reshape((2, 2, 2, 2)).requires_grad_()
    # l = QuantizeBias(2)
    # l.bias.data.fill_(1)
    # l.last_value.data[0] = 15.
    # l.eval()
    # print(l)
    # print(l(a))
    # s = torch.sum(l(a))
    # s.backward()
    # print(a.grad)
    # print(l.bias.grad)
