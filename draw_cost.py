import os
import skimage.transform
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
plt.rcParams.update({'figure.max_open_warning': 0})
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams.update({'font.size': 16})

import numpy as np

costs = []
base_dir = '/data/zhijie/snapshots_ab_in/'
dir_list = ['in-ft-mf--single_eps', 'in-ft-mf--local_reparam', 'in-ft-mf--flipout', 'in-ft-mf--er']
for dir in dir_list:
    cost = []
    with open(os.path.join(base_dir, dir, 'cost.txt')) as f:
        lines = f.readlines()
        for line in lines:
            line = line.split(" ")
            cost.append([float(line[0]), float(line[1])])
    cost = np.array(cost)
    print(cost.shape)
    costs.append(cost)
costs = np.stack(costs)


lw = 1.25
color = ['red', 'green', 'darkorange', 'b']
# rets = np.load(os.path.join(dir_, 'ens_rets.npy'))
# if isinstance(rets, list):
#     rets = np.stack([np.array(item) for item in rets])
# min_acc = min(rets[:, 2].min(), rets[:, 6].min(), baseline_acc) - 0.1
# max_acc = max(rets[:, 2].max(), rets[:, 6].max(), baseline_acc) + 0.1

fig = plt.figure(figsize=(4,3))
fig, ax1 = plt.subplots(figsize=(4,3))
l1 = ax1.plot(costs[0][:, 0], costs[0][:, 1], color=color[0], lw=lw, alpha=0.6)
l2 = ax1.plot(costs[1][:, 0], costs[1][:, 1], color=color[1], lw=lw)
l3 = ax1.plot(costs[2][:, 0], costs[2][:, 1], color=color[2], lw=lw, alpha=0.6)
l4 = ax1.plot(costs[3][:, 0], costs[3][:, 1], color=color[3], lw=lw)
ax1.set_yticks(np.arange(0.9, 1.4, 0.1))
ax1.set_xticks(list(np.arange(1, costs[:, :, 0].max()+600, 1800)))
ax1.set_xticklabels([int(i) for i in list(np.arange(0, costs[:, :, 0].max()/60.+10, 30))])
ax1.set_ylim((0.9, 1.4))
ax1.set_xlim((1, costs[:, :, 0].max()+300))
ax1.set_xlabel('Time (min)')
ax1.set_ylabel('xx')
# ax2 = ax1.twinx()
# l3 = ax2.plot(rets[:, 0]+1, rets[:, 4], color=color[2], lw=lw, alpha=0.6)
# l4 = ax2.plot(rets[:, 0]+1, rets[:, 8], color=color[3], lw=lw)
# ax2.set_ylabel('ECE')
# ax2.set_ylim((0.0, max_mi))
# ax2.set_yticks(np.arange(0.0, max_mi+1e-6, max_mi/4.))
ax1.legend(l1+l2+l3+l4, ['Vanilla', 'Local', 'Flipout', 'ER'], loc = 'best', fancybox=True, columnspacing=0.5, handletextpad=0.2, borderpad=0.15) # +l3+l4 , 'Indiv ECE', 'Ensemble ECE'  , fontsize=11
plt.savefig(os.path.join('cost_plot.pdf'), format='pdf', dpi=600, bbox_inches='tight')
