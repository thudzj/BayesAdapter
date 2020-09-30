import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
plt.rcParams.update({'figure.max_open_warning': 0})
plt.rcParams['font.family'] = 'Times New Roman'
# plt.rcParams.update({'font.size': 20})

import numpy as np


mis = np.load('/data/zhijie/snapshots_ab_in/ft-gan-.75-1-alpha3/mis_tmp.npy')
accs = np.load('/data/zhijie/snapshots_ab_in/ft-gan-.75-1-alpha3/accs_tmp.npy')

if False:
    print(accs.mean(), mis.shape, accs.shape, mis.min(), mis.max())
    in_bin = (mis < 0.45).astype(np.int32)
    print(in_bin.astype(np.float32).mean(), accs[in_bin.astype(bool)].astype(np.float32).mean())
else:
    bin_boundaries_plot = []
    for i in range(11):
        bin_boundaries_plot.append(np.percentile(mis, i*10))
    bin_lowers_plot = bin_boundaries_plot[:-1]
    bin_uppers_plot = bin_boundaries_plot[1:]

    accuracy_in_bin_list = []
    for bin_lower, bin_upper in zip(bin_lowers_plot, bin_uppers_plot):
        in_bin = (mis > bin_lower).astype(np.int32) * (mis <= bin_upper).astype(np.int32)
        prop_in_bin = in_bin.astype(np.float32).mean()
        accuracy_in_bin = 0
        if prop_in_bin > 0:
            accuracy_in_bin = accs[in_bin.astype(bool)].astype(np.float32).mean()
        print(prop_in_bin, accuracy_in_bin)
        accuracy_in_bin_list.append(accuracy_in_bin)

    fig = plt.figure(figsize=(6,4.5))
    p1 = plt.bar(np.arange(10) / 10. * (mis.max()), accuracy_in_bin_list, (mis.max())/10., align = 'edge', edgecolor ='black')
    # p2 = plt.plot([0,1], [0,1], '--', color='gray')

    plt.ylabel('Accuracy', fontsize=20)
    plt.xlabel('Uncertainty', fontsize=20)
    #plt.title(title)

    plt.gca().set_xticks(np.arange(11) / 10. * (mis.max()))
    plt.gca().set_xticklabels(['{:.1e}'.format(i) for i in bin_boundaries_plot], fontsize=9)

    # plt.xticks(bin_boundaries_plot)
    # for i, j in enumerate(bin_boundaries_plot):
    #     plt.gca().get_xaxis().get_offset_text().set_position((mis.max()/10.*i,j))
    plt.yticks(np.arange(0, 1.01, 0.2), fontsize=16)
    plt.xlim(left=0,right=mis.max())
    plt.ylim(bottom=0,top=1)
    # plt.grid(True)
    #plt.legend((p1[0], p2[0]), ('Men', 'Women'))
    # plt.text(0.1, 0.83, 'ECE: {:.4f}'.format(ece.item()), fontsize=14)
    print(1)
    plt.savefig("rej_dec.pdf", format='pdf', dpi=600, bbox_inches='tight')
    print(2)
