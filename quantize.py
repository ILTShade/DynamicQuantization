#!/home/sunhanbo/software/anaconda3/bin/python
#-*-coding:utf-8-*-
import pdb
import torch
import torch.nn as nn
from torch.autograd import Function
import torch.nn.functional as F
import math
import copy

# power
power_scale = 1

# 在本次的实验假设中，对输入定点，对权重定点，对输出定点
# 但是不要求前一层的输出一定是下一层的输入，因此不需要在这里的逻辑关系
# 不需要保留定点的系数，保留相关结果即可
# 本工程对权重的定点不做严格要求，期望能够通过一种定点训练方法找到动态的量化范围
# quantize Function
class QuantizeFunction(Function):
    @staticmethod
    def forward(ctx, input, fix_config, training, last_value = None):
        # last_value是上一次的最大值矩阵，training是训练的标志位
        # fix_config是定点的相关配置，是一个字典，包含mode, qbit, ratio等参数
        # 如果是训练，那么last_value中的值要发生变化，否则以last_value中的值为准
        # weight的定点位置可以随着最大值而改变，但是activation不行，因为不同的图片activation不同
        # scale
        if fix_config['mode'] == 'weight':
            scale = torch.max(torch.abs(input))
        elif fix_config['mode'] == 'activation':
            if training:
                momentum = fix_config['momentum']
                last_value.data[0] = momentum * last_value.item() + (1 - momentum) * torch.max(torch.abs(input)).item()
            scale = last_value.item()
        else:
            raise NotImplementedError
        # transfer info
        qbit = fix_config['qbit']
        ratio_L = fix_config['ratio_L']
        ratio_H = fix_config['ratio_H']
        thres = 2 ** (qbit - 1) - 1
        x_range = thres / (ratio_H - ratio_L)
        global power_scale
        power_scale = power_scale * (scale / x_range)
        # transfer
        output = input / scale
        output_sign = torch.sign(output)
        output_value = torch.abs(output)
        output_value[output_value > ratio_H] = 1
        output_value[output_value < ratio_L] = 0
        return torch.round(output_value * x_range) * output_sign * scale / x_range
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None, None, None, None
Quantize = QuantizeFunction.apply

# 量化的卷积层，定点参数需要每层指定
class QuantizePowerConv2d(nn.Module):
    def __init__(self, fix_config_dict, in_channels, out_channels, kernel_size, \
                 stride = 1, padding = 0, dilation = 1, groups = 1, bias = True):
        super(QuantizePowerConv2d, self).__init__()
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.last_value_input = nn.Parameter(torch.zeros((1)))
        self.last_value_output = nn.Parameter(torch.zeros((1)))
        self.input_fix_config = fix_config_dict['input']
        self.weight_fix_config = fix_config_dict['weight']
        self.output_fix_config = fix_config_dict['output']
    def forward(self, x):
        # dynamic quantize
        global power_scale
        power_scale = 1
        if self.input_fix_config['mode'] == 'input':
            quantize_input = x
        else:
            quantize_input = Quantize(x,
                                      self.input_fix_config,
                                      self.training,
                                      self.last_value_input,
                                      )
        quantize_weight = Quantize(self.conv2d.weight,
                                   self.weight_fix_config,
                                   self.training,
                                   None,
                                   )
        output = F.conv2d(quantize_input,
                          quantize_weight,
                          self.conv2d.bias,
                          self.conv2d.stride,
                          self.conv2d.padding,
                          self.conv2d.dilation,
                          self.conv2d.groups,
                          )
        # power
        if self.training:
            norm_x = quantize_input / torch.mean(torch.abs(quantize_input))
            norm_w = quantize_weight / torch.mean(torch.abs(quantize_weight))
            square_sum = F.conv2d(torch.mul(norm_x, norm_x), \
                                  torch.abs(norm_w), \
                                  None, \
                                  self.conv2d.stride, \
                                  self.conv2d.padding, \
                                  self.conv2d.dilation, \
                                  self.conv2d.groups, \
                                  )
        else:
            square_sum = F.conv2d(torch.mul(quantize_input, quantize_input), \
                                  torch.abs(quantize_weight), \
                                  None, \
                                  self.conv2d.stride, \
                                  self.conv2d.padding, \
                                  self.conv2d.dilation, \
                                  self.conv2d.groups, \
                                  )
            square_sum = square_sum / power_scale
        quantize_output = Quantize(output,
                                   self.output_fix_config,
                                   self.training,
                                   self.last_value_output,
                                   )
        return quantize_output, torch.sum(square_sum)
