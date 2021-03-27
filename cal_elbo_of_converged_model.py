# calculate the elbos of converged models (CIFAR-10)
# on gpu33
# loc   ckpt                                        nll loss      kl loss
# 33    /data/zhijie/snapshots_ab/train-mf-1        0.0318        2384.3250
# 33    /data/zhijie/snapshots_ab/ft-mf-mc20-1      0.0191        2806.8403
import torch, os
import numpy as np
prior_mu = 0
prior_sigma2 = 1. / 50000 / 0.0002
path = '/data/zhijie/snapshots_ab/ft-mf-mc20-2' #'/data/zhijie/snapshots_ab/train-mf-1' #

ckpt = torch.load(os.path.join(path, 'checkpoint.pth.tar'), map_location='cpu')['state_dict']

params = {}
for k, v in ckpt.items():
    if 'num_batches_tracked' in k or 'running_mean' in k or 'running_var' in k:
        continue
    params[k] = v

print(len(params))
cnt = 0
kl = 0
for k, v in params.items():
    if '_mu' in k and v.dim() != 1:
        cnt += 1
        v_psi = params[k.replace('mu', 'psi')]
        v_sigma2 = v_psi.mul(2).exp()
        kl1 = -0.5*(1 + v_sigma2.log() - np.log(prior_sigma2) - v**2/prior_sigma2 - v_sigma2/prior_sigma2).sum()/50000
        kl2 = (v**2 + v_sigma2).sum()*0.5*0.0002 - v_psi.mul(2).sum()/2/50000
        print(k)

        kl += kl1
        # print(k, kl)

print(cnt, kl)
