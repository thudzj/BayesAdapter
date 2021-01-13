import torch
import numpy as np
weights = torch.load('/data/zhijie/snapshots_ab_in/ft-gan-.75-1-alpha3/checkpoint.pth.tar', map_location='cpu')['state_dict']
mean = weights['module.conv1.weight_mu'].permute(0, 2, 3, 1).data.cpu().numpy()
sigma = weights['module.conv1.weight_log_sigma'].mul(2).exp().permute(0, 2, 3, 1).data.cpu().numpy()
print(mean.shape, sigma.shape)

mean = (mean - mean.min())
sigma /= sigma.max()
mean /= mean.max()


import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
plt.rcParams.update({'figure.max_open_warning': 0})
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams.update({'font.size': 16})

# settings
figsize = [16, 8]     # figure size, inches

# prep (x,y) for extra plotting on selected sub-plots

bigimg = np.zeros((63, 126, 3))
for i in range(8):
    for j in range(8):
        bigimg[i*8:i*8+7, j*8:j*8+7, :] = mean[i*8+j]

    for j in range(8):
        bigimg[i*8:i*8+7, j*8+63:j*8+7+63, :] = sigma[i*8+j]

# create figure (fig), and array of axes (ax)
fig, ax = plt.subplots()#( figsize=figsize)

ax.imshow(bigimg, cmap=plt.cm.BrBG, interpolation='nearest', origin='lower', extent=[0,1,0,1],aspect="auto")
ax.set_xticks([])
ax.set_aspect(1/2.)
ax.set_yticks([])
plt.tight_layout(True)
plt.savefig('posterior.pdf')

exit()
# plot simple raster image on each sub-plot
for i, axi in enumerate(ax.flat):
        # i runs from 0 to (nrows*ncols-1)
        rowid = i // ncols
        colid = i % ncols

        if colid < 8:
            img = mean[rowid*8+colid]
        else:
            img = sigma[rowid*8+colid-8]
        axi.imshow(img, cmap=plt.cm.BrBG, interpolation='nearest', origin='lower')#, extent=[0,1,0,1])
        axi.set_xticks([])
        axi.set_yticks([])
        #axi.set_title("Row:"+str(rowid)+", Col:"+str(colid))

plt.tight_layout(True)
plt.savefig('posterior.pdf')
