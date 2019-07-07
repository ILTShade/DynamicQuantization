#!/home/sunhanbo/software/anaconda3/bin/python
#-*-coding:utf-8-*-
import subprocess
import time
import os

NetClass = ['lenet', 'vgg', 'resnet']
ActivationRange = [8, 6, 4]
WeightRange = [8, 6, 4]

TaskList = []
for net_class in NetClass:
    # float
    PREFIX = 'PA_%s_BASELINE' % net_class
    CMD = './synthesize.py -g %%d -d cifar10 -n %s -t train_wo_power -p %s 2>&1 > %s.log.txt &' % (net_class + '_float', PREFIX, PREFIX)
    TaskList.append((CMD, PREFIX))
    net_module = net_class + '_quantize'
    for activation_bit in ActivationRange:
        for weight_bit in WeightRange:
            PREFIX = 'PA_%s_MAX_A%dW%d' % (net_class, activation_bit, weight_bit)
            if activation_bit == 8 and weight_bit == 8:
                CMD = './synthesize.py -g %%d -d cifar10 -n %s -t train_wo_power -ab 8 -wb 8 -p %s' % (net_module, PREFIX)
            else:
                CMD = './synthesize.py -g %%d -d cifar10 -n %s -t finetune_wo_power -ab %d -wb %d -w zoo/PA_%s_MAX_A8W8_params.pth -p %s 2>&1 > %s.log.txt &' % \
                      (net_module, activation_bit, weight_bit, net_class, PREFIX, PREFIX)
            CMD += (' 2>&1 > %s.log.txt &' % PREFIX)
            TaskList.append((CMD, PREFIX))

TaskWait = [-1, -1] + [1]*8 + [-1, -1] + [11]*8 + [-1, -1] + [21]*8
TaskReady = [0]*len(TaskList)

OccupyList = ['', '', '', '']
while True:
    # 检查状态
    for i, occupy in enumerate(OccupyList):
        if occupy != '':
            run_status = subprocess.getoutput('ps -ef | grep %s' % occupy).split('\n')
            if len(run_status) <= 2:
                for j, task in enumerate(TaskList):
                    if task[1] == occupy:
                        print('end %s' % task[1])
                        TaskReady[j] = 1
                        break
                OccupyList[i] = ''
    # 状态更新后查看空闲GPU
    gpu_id = -1
    for i, occupy in enumerate(OccupyList):
        if occupy == '':
            gpu_id = i
            break
    # 判断是否满足条件
    task_id = -1
    if gpu_id != -1:
        for i, task in enumerate(TaskList):
            if TaskReady[i] == 0:
                if TaskWait[i] == -1:
                    task_id = i
                    break
                elif TaskReady[TaskWait[i]] == 1:
                    task_id = i
                    break
    # 增加一个进程上去
    if task_id != -1:
        CMD = TaskList[task_id][0] % gpu_id
        print('start %s' % TaskList[task_id][1])
        print(CMD)
        os.system(CMD)
        TaskReady[task_id] = 0.5
        OccupyList[gpu_id] = TaskList[task_id][1]
    # 结束判定
    S = sum(TaskReady)
    if S == len(TaskReady):
        break
    time.sleep(10)
