#!/home/sunhanbo/software/anaconda3/bin/python
#-*-coding:utf-8-*-
import torch
import numpy as np
f = open('./zoo/data_analysis.txt', 'w')

DATA = torch.load('./zoo/DATA.pt')
D = []
for i in range(7):
    D.append(torch.cat([DATA[j][i] for j in range(100)], dim = 0))

max_value = []
for i in range(7):
    max_value.append(torch.max(torch.abs(D[i])))

scale = [1, 1./2, 1./4, 1./8, 1./16]
f.write('data distribution\n')
for s in scale:
    r = []
    for i in range(7):
        ratio = float(torch.sum(torch.abs(D[i]) <= s*max_value[i]).item()) / D[i].numel()
        r.append(ratio)
    f.write('%f: %s' % (s, ' '.join([str(v) for v in r])) + '\n')

r = []
for i in range(7):
    ratio = float(torch.sum(torch.abs(D[i]) <= 3*torch.std(D[i])).item()) / D[i].numel()
    r.append(ratio)
f.write('3sigma: %s' % (' '.join([str(v) for v in r])) + '\n')

# 这部分计算由量化带来的量化误差
f.write('data quantize\n')
T = torch.clamp(D[0]/(3*torch.std(D[0])), -1, 1).numpy().astype(np.float64)
T = T.flatten()
np.random.shuffle(T)
T = T[0:int(len(T)/100)]
k = 4
B = np.round(T*(2**(k-1)-1))/(2**(k-1)-1)
E = np.abs(T-B)
hist, bin_edges = np.histogram(E, 50, (0,0.2))
f.write('standard quantization\n')
f.write('%f\n' % np.mean(E))
f.write('%s\n' % (' '.join([str(v) for v in hist])))

for r in [3., 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7, 3.8, 3.9]:
    B1 = 1.0 / (1.0 + (np.exp(-r*T)))
    B2 = np.round(B1*(2**k-2))/(2**k-2)
    B3 = np.log((1.0/B2)-1)/(-r)
    B4 = np.where(B2 == 0, -1, B3)
    B5 = np.where(B2 == 1, 1, B4)
    E = np.abs(T-B5)
    hist, bin_edges = np.histogram(E, 50, (0, 0.2))
    f.write('sigmoid scale %f\n' % r)
    f.write('%f\n' % np.mean(E))
    f.write('%s\n' % (' '.join([str(v) for v in hist])))
f.close()
