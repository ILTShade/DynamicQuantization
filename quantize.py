#!/home/sunhanbo/software/anaconda3/bin/python
#-*-coding:utf-8-*-
import torch
import torch.nn as nn
from torch.autograd import Function
import torch.nn.functional as F

# power scale record
power_scale = 1
# 在本次的实验假设中，对输入定点，对权重定点，对输出定点
# 但是不要求前一层的输出一定是下一层的输入
# 不需要保留定点的系数，保留相关结果即可
# 本工程对权重的定点不做严格要求，期望能够通过一种定点训练方法找到动态的量化范围
# quantize Function
class QuantizeFunction(Function):
    @staticmethod
    def forward(ctx, input, fix_config, training, last_value = None):
        # 对于不同的模式，采用完全不同的量化方法
        global power_scale
        if fix_config['mode'] == 'input':
            # 此部分只对整个网络的输入做变换，数据范围是[0,1]
            power_scale = 1
            return input
        elif fix_config['mode'] == 'activation_in':
            # 此部分对输入的激活做变换，默认输入的激活均为非负数，这样可以忽略负数的影响
            if training:
                momentum = fix_config['momentum']
                last_value.data[0] = momentum * last_value.item() + (1 - momentum) * torch.max(torch.abs(input)).item()
            scale = last_value.item()
            thres = 2 ** (fix_config['qbit']) - 1
            output = torch.div(input, scale)
            power_scale = scale
            return output.clamp_(0, 1).mul_(thres).round_().div(thres/scale)
        elif fix_config['mode'] == 'weight':
            # 此部分对权重做变换，直接采用最大值，可以尽可能少地产生误差，网络权重有正有负
            scale = torch.max(torch.abs(input))
            thres = 2 ** (fix_config['qbit'] - 1) - 1
            output = torch.div(input, scale)
            power_scale *= scale
            return output.clamp_(-1, 1).mul_(thres).round_().div(thres/scale)
        elif fix_config['mode'] == 'activation_out':
            # 此部分对输出的激活做变换，输出的激活有正有负，可能需要采用非均匀量化的方式
            # 这部分采用3sigma原则来对范围进行限制
            # 不需要对power_scale进行处理
            if training:
                momentum = fix_config['momentum']
                last_value.data[0] = momentum * last_value.item() + (1 - momentum) * torch.std(input).item()
            scale = 3 * last_value.item()
            output = torch.div(input, scale)
            # thres = 2 ** (fix_config['qbit'] - 1) - 1
            # return output.clamp_(-1, 1).mul_(thres).round_().div(thres/scale)
            ratio = 1
            thres = 2 ** (fix_config['qbit'])
            return output.mul_(ratio).sigmoid_().mul_(thres).round_().clamp_(1, thres - 1).div_(thres).reciprocal_().sub_(1).log_().div(-ratio/scale)
        else:
            raise NotImplementedError
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None, None, None, None
Quantize = QuantizeFunction.apply

# 量化的卷积层，不统计每层的crossbar能耗
class QuantizeConv2d(nn.Module):
    def __init__(self, fix_config_dict, in_channels, out_channels, kernel_size, \
                 stride = 1, padding = 0, dilation = 1, groups = 1, bias = True):
        super(QuantizeConv2d, self).__init__()
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.last_value_input = nn.Parameter(torch.ones((1)))
        self.last_value_output = nn.Parameter(torch.ones((1)))
        self.input_fix_config = fix_config_dict['input']
        self.weight_fix_config = fix_config_dict['weight']
        self.output_fix_config = fix_config_dict['output']
    def forward(self, x):
        # dynamic quantize
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
        quantize_output = Quantize(output,
                                   self.output_fix_config,
                                   self.training,
                                   self.last_value_output,
                                   )
        return quantize_output
    def extra_repr(self):
        extra_def = []
        for name, config in zip(['input', 'weight', 'output'], [self.input_fix_config, self.weight_fix_config, self.output_fix_config]):
            part_def = name
            for key, value in config.items():
                part_def = part_def + ' ' + key + ':' + str(value) + ','
            extra_def.append(part_def)
        return '\n'.join(extra_def)

# 量化的卷积层，统计每层的crossbar能耗
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
        quantize_output = Quantize(output,
                                   self.output_fix_config,
                                   self.training,
                                   self.last_value_output,
                                   )
        # power
        if self.training:
            norm_x = quantize_input / torch.mean(torch.abs(quantize_input))
            norm_w = quantize_weight / torch.mean(torch.abs(quantize_weight))
            square_sum = F.conv2d(torch.mul(norm_x, norm_x),
                                  torch.abs(norm_w),
                                  None,
                                  self.conv2d.stride,
                                  self.conv2d.padding,
                                  self.conv2d.dilation,
                                  self.conv2d.groups,
                                  )
        else:
            square_sum = F.conv2d(torch.mul(quantize_input, quantize_input),
                                  torch.abs(quantize_weight),
                                  None,
                                  self.conv2d.stride,
                                  self.conv2d.padding,
                                  self.conv2d.dilation,
                                  self.conv2d.groups,
                                  )
            square_sum = square_sum / real_power_scale
        return quantize_output, torch.sum(square_sum)
    def extra_repr(self):
        extra_def = []
        for name, config in zip(['input', 'weight', 'output'], [self.input_fix_config, self.weight_fix_config, self.output_fix_config]):
            part_def = name
            for key, value in config.items():
                part_def = part_def + ' ' + key + ':' + str(value) + ','
            extra_def.append(part_def)
