#!/home/sunhanbo/software/anaconda3/bin/python
#-*-coding:utf-8-*-
import quantize
import torch
from importlib import import_module
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('-g', '--gpu', help = 'select gpu')
parser.add_argument('-d', '--dataset', help = 'select dataset')
parser.add_argument('-n', '--net', help = 'select net')
parser.add_argument('-t', '--train', help = 'select train')
parser.add_argument('-p', '--prefix', help = 'select prefix')
parser.add_argument('-w', '--weight', help = 'select weight model')
parser.add_argument('-wb', '--weight_bit', help = 'select weight bit')
parser.add_argument('-ab', '--activation_bit', help = 'select activation bit')
args = parser.parse_args()
assert args.gpu
assert args.dataset
assert args.net
assert args.train
assert args.prefix

# 相关参数的选择
# 如果是finetune，那么需要选择定点模型
finetune_flag = False
if 'finetune' in args.train:
    finetune_flag = True
    assert args.weight
    print('finetune from %s' % args.weight)
print('train from none')
# 如果是定点模型，那么需要选择定点位宽
quantize_flag = False
if 'quantize'in args.net:
    quantize_flag = True
    assert args.weight_bit
    assert args.activation_bit
    print('fix model activation bit %s weight bit %s' % (args.activation_bit, args.weight_bit))
print('float model')

# dataloader
dataset_module = import_module(args.dataset)
train_loader, test_loader = dataset_module.get_dataloader()
# net
net_module = import_module(args.net)
if quantize_flag:
    net_module.activation_bit = int(args.activation_bit)
    net_module.weight_bit = int(args.weight_bit)
net = net_module.get_net()
# train
train_module = import_module(args.train)
device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
if finetune_flag:
    train_module.train_net(net, train_loader, test_loader, 'train', device, args.prefix, args.weight)
else:
    train_module.train_net(net, train_loader, test_loader, 'train', device, args.prefix)
