#!/home/sunhanbo/software/anaconda3/bin/python
#-*-coding:utf-8-*-
import torch
import torch.nn as nn
from torch.autograd import Function
import torch.nn.functional as F

# power scale record
power_scale = 1
real_power_scale = 1

# 在本次的实验假设中，对输入定点，对权重定点，对输出定点
# 但是不要求前一层的输出一定是下一层的输入
# 不需要保留定点的系数，保留相关结果即可
# 本工程对权重的定点不做严格要求，期望能够通过一种定点训练方法找到动态的量化范围
# quantize Function
class QuantizeFunction(Function):
    @staticmethod
    def forward(ctx, input, fix_config, training, last_value = None):
        # last_value是上一次的最大值，training是训练的标志位
        # fix_config是定点的相关配置，是一个字典，包含mode, qbit, ratio等参数
        # 如果是训练，那么last_value中的值要发生变化，否则以last_value中的值为准
        # weight的定点位置可以随着最大值而改变，但是activation不行，因为不同的输入activation不同
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
        ratio = fix_config['ratio']
        thres = 2 ** (qbit - 1) - 1
        scale = scale * ratio
        global power_scale
        power_scale = power_scale * scale
        # transfer
        return output.div_(scale).clamp_(-1, 1).mul_(thres).round_().div_(thres).mul_(scale)
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
        global power_scale
        global real_power_scale
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
        real_power_scale = power_scale
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
        global power_scale
        global real_power_scale
        if self.input_fix_config['mode'] == 'input':
            power_scale = 1
            quantize_input = x
        else:
            power_scale = 1
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
        real_power_scale = power_scale
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
