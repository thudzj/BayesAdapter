from __future__ import division
import os, sys, shutil, time, random, math
import argparse
import warnings
import numpy as np

import torch
import torch.backends.cudnn as cudnn

import torch.nn.parallel
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.utils.data.distributed

import models
from utils import AverageMeter, RecorderMeter, time_string, convert_secs2time, _ECELoss, load_dataset_ft, plot_mi, plot_ens, ent
from mean_field import *

model_names = sorted(name for name in models.__dict__ if name.islower() and not name.startswith("__") and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='Training script for CIFAR', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

# Data / Model
parser.add_argument('--data_path', metavar='DPATH', default='/data/zhijie/data', type=str, help='Path to dataset') #/data/LargeData/cifar/
parser.add_argument('--dataset', metavar='DSET', default='cifar10', type=str, choices=['cifar10', 'cifar100', 'imagenet'], help='Choose between CIFAR/ImageNet.')
parser.add_argument('--arch', metavar='ARCH', default='wrn', help='model architecture: ' + ' | '.join(model_names) + ' (default: wrn)')
parser.add_argument('--depth', type=int, metavar='N', default=28)
parser.add_argument('--wide', type=int, metavar='N', default=10)

parser.add_argument('--save_path', type=str, default='/data/zhijie/snapshots_ab/', help='Folder to save checkpoints and log.')
parser.add_argument('--ckpts', type=str, nargs='+', default=['map-decay2', 'map-decay2-1', 'map-decay2-2', 'map-decay2-3', 'map-decay2-4'])
parser.add_argument('--manualSeed', type=int, default=0, help='manual seed')

# Optimization
parser.add_argument('--batch_size', type=int, default=32, help='Batch size.')

#Regularization
parser.add_argument('--cutout', dest='cutout', action='store_true', help='Enable cutout augmentation')

# Acceleration
parser.add_argument('--workers', type=int, default=4, help='number of data loading workers (default: 2)')

# attack settings
parser.add_argument('--epsilon', default=0.031, type=float,
                    help='perturbation')
parser.add_argument('--epsilon_scale', default=1., type=float)
parser.add_argument('--num-steps', default=20, type=int,
                    help='perturb number of steps')
parser.add_argument('--step-size', default=0.003, type=float,
                    help='perturb step size')
parser.add_argument('--random',
                    default=True,
                    help='random initialization for PGD')

parser.add_argument('--max_choice', type=int, default=None)
parser.add_argument('--num_gan', type=int, default=10)
parser.add_argument('--blur_prob', type=float, default=0.5)
parser.add_argument('--blur_sig', type=float, nargs='+', default=[0., 3.])
parser.add_argument('--jpg_prob', type=float, default=0.5)
parser.add_argument('--jpg_method', type=str, nargs='+', default=['cv2', 'pil'])
parser.add_argument('--jpg_qual', type=int, nargs='+', default=[30, 100])

best_acc = 0

def main():
    args = parser.parse_args()
    args.bayes = None
    args.dropout_rate = 0
    args.distributed = False
    if not os.path.isdir(args.data_path): os.makedirs(args.data_path)
    job_id = "deep_ensemble"
    args.save_path = args.save_path + job_id
    if not os.path.isdir(args.save_path): os.makedirs(args.save_path)
    args.num_classes = 10 if args.dataset == 'cifar10' else 100
    args.num_data = 50000

    args.use_cuda = torch.cuda.is_available()
    if args.manualSeed is None: args.manualSeed = random.randint(1, 10000)
    random.seed(args.manualSeed)
    np.random.seed(args.manualSeed)
    torch.manual_seed(args.manualSeed)
    if args.use_cuda: torch.cuda.manual_seed_all(args.manualSeed)
    cudnn.deterministic = True

    main_worker(0, 1, args)

def main_worker(gpu, ngpus_per_node, args):
    global best_acc
    args.gpu = gpu
    assert args.gpu is not None
    print("Use GPU: {} for training".format(args.gpu))

    nets = []
    for ckpt in args.ckpts:
        net = models.__dict__[args.arch](args, args.depth, args.wide, args.num_classes)
        state_dict = torch.load(os.path.join('/data/zhijie/snapshots_ab', ckpt, 'checkpoint.pth.tar'), map_location='cpu')['state_dict']
        net.load_state_dict({k.replace('module.', ''): v for k,v in state_dict.items()})
        nets.append(net)

    torch.cuda.set_device(args.gpu)
    for net in nets:
        net = net.cuda(args.gpu)
    criterion = torch.nn.CrossEntropyLoss().cuda(args.gpu)
    cudnn.benchmark = True

    train_loader, train_loader1, test_loader, test_loader1, fake_loader, fake_loader2 = load_dataset_ft(args)
    evaluate(test_loader, test_loader1, fake_loader, fake_loader2, nets, criterion, args)

def evaluate(test_loader, test_loader1, fake_loader, fake_loader2, nets, criterion, args):
    ens_validate(test_loader, nets, criterion, args)

    ens_attack(test_loader1, nets, criterion, args)
    print('NAT vs. ADV: AP {}'.format(plot_mi(args.save_path, 'advg')))

    ens_validate(fake_loader, nets, criterion, args, suffix='_fake')
    print('NAT vs. Fake (SNGAN): AP {}'.format(plot_mi(args.save_path, 'fake')))

    ens_validate(fake_loader2, nets, criterion, args, suffix='_fake2')
    print('NAT vs. Fake (PGGAN): AP {}'.format(plot_mi(args.save_path, 'fake2')))

def ens_validate(val_loader, models, criterion, args, suffix=''):
    num_ens = len(models)
    for model in models:
        model.eval()

    ece_func = _ECELoss().cuda(args.gpu)
    with torch.no_grad():
        targets = []
        mis = [0 for _ in range(len(val_loader))]
        preds = [0 for _ in range(len(val_loader))]
        rets = torch.zeros(num_ens, 9).cuda(args.gpu)
        for i, (input, target) in enumerate(val_loader):
            input = input.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)
            targets.append(target)
            for ens in range(num_ens):
                output = models[ens](input)

                one_loss = criterion(output, target)
                one_prec1, one_prec5 = accuracy(output, target, topk=(1, 5))

                mis[i] = (mis[i] * ens + (-output.softmax(-1) * output.log_softmax(-1)).sum(1)) / (ens + 1)
                preds[i] = (preds[i] * ens + output.softmax(-1)) / (ens + 1)

                loss = criterion(preds[i].log(), target)
                prec1, prec5 = accuracy(preds[i], target, topk=(1, 5))

                rets[ens, 0] += ens*target.size(0)
                rets[ens, 1] += one_loss.item()*target.size(0)
                rets[ens, 2] += one_prec1.item()*target.size(0)
                rets[ens, 3] += one_prec5.item()*target.size(0)
                rets[ens, 5] += loss.item()*target.size(0)
                rets[ens, 6] += prec1.item()*target.size(0)
                rets[ens, 7] += prec5.item()*target.size(0)

        preds = torch.cat(preds, 0)
        confidences, predictions = torch.max(preds, 1)
        targets = torch.cat(targets, 0)
        mis = (- preds * preds.log()).sum(1) - (0 if num_ens == 1 else torch.cat(mis, 0))
        rets /= targets.size(0)

        rets = rets.data.cpu().numpy()
        if suffix == '':
            ens_ece = ece_func(confidences, predictions, targets, os.path.join(args.save_path, 'ens_cal{}.pdf'.format(suffix)))
            rets[-1, -1] = ens_ece
            print("nll {:.4f} acc1 {:.4f} acc5 {:.4f} ece {:.4f}".format(rets[-1, 5], rets[-1, 6], rets[-1, 7], rets[-1, 8]))

    if args.gpu == 0:
        np.save(os.path.join(args.save_path, 'mis{}.npy'.format(suffix)), mis.data.cpu().numpy())
    return rets

def ens_attack(val_loader, models, criterion, args):
    def _grad(X, y, mean, std):
        probs = torch.zeros(num_ens, X.shape[0]).cuda(args.gpu)
        grads = torch.zeros(num_ens, *list(X.shape)).cuda(args.gpu)
        for j in range(num_ens):
            with torch.enable_grad():
                X.requires_grad_()
                output = models[j](X.sub(mean).div(std))
                loss = torch.nn.functional.cross_entropy(output, y, reduction='none')
                grad_ = torch.autograd.grad(
                    [loss], [X], grad_outputs=torch.ones_like(loss), retain_graph=False)[0].detach()
            grads[j] = grad_
            probs[j] = torch.gather(output.detach().softmax(-1), 1, y[:,None]).squeeze()
        probs /= probs.sum(0)
        grad_ = (grads * probs[:, :, None, None, None]).sum(0)
        return grad_

    def _pgd_whitebox(X, y, mean, std):
        X_pgd = X.clone()
        if args.random: X_pgd += torch.cuda.FloatTensor(*X_pgd.shape).uniform_(-args.epsilon, args.epsilon)

        for _ in range(args.num_steps):
            grad_ = _grad(X_pgd, y, mean, std)
            X_pgd += args.step_size * grad_.sign()
            eta = torch.clamp(X_pgd - X, -args.epsilon, args.epsilon)
            X_pgd = torch.clamp(X + eta, 0, 1.0)

        mis = 0
        preds = 0
        for ens in range(num_ens):
            output = models[ens](X_pgd.sub(mean).div(std))
            mis = (mis * ens + (-output.softmax(-1) * (output).log_softmax(-1)).sum(1)) / (ens + 1)
            preds = (preds * ens + output.softmax(-1)) / (ens + 1)

        loss = criterion((preds+1e-8).log(), target)
        prec1, prec5 = accuracy(preds, target, topk=(1, 5))
        mis = (- preds * (preds+1e-8).log()).sum(1) - (0 if num_ens == 1 else mis)
        return loss, prec1, prec5, mis

    if args.dataset == 'cifar10':
        mean = torch.from_numpy(np.array([x / 255 for x in [125.3, 123.0, 113.9]])).view(1,3,1,1).cuda(args.gpu).float()
        std = torch.from_numpy(np.array([x / 255 for x in [63.0, 62.1, 66.7]])).view(1,3,1,1).cuda(args.gpu).float()
    elif args.dataset == 'cifar100':
        mean = torch.from_numpy(np.array([x / 255 for x in [129.3, 124.1, 112.4]])).view(1,3,1,1).cuda(args.gpu).float()
        std = torch.from_numpy(np.array([x / 255 for x in [68.2, 65.4, 70.4]])).view(1,3,1,1).cuda(args.gpu).float()

    losses, top1, top5 = 0, 0, 0
    num_ens = len(models)
    for model in models:
        model.eval()
    with torch.no_grad():
        mis = []
        for i, (input, target) in enumerate(val_loader):
            input = input.cuda(args.gpu, non_blocking=True).mul_(std).add_(mean)
            target = target.cuda(args.gpu, non_blocking=True)
            loss, prec1, prec5, mis_ = _pgd_whitebox(input, target, mean, std)
            losses += loss * target.size(0)
            top1 += prec1 * target.size(0)
            top5 += prec5 * target.size(0)
            mis.append(mis_)

        mis = torch.cat(mis, 0)
        losses /= mis.size(0)
        top1 /= mis.size(0)
        top5 /= mis.size(0)

    print('ADV ensemble TOP1: {:.4f}, TOP5: {:.4f}, LOS: {:.4f}'.format(top1.item(), top5.item(), losses.item()))
    if args.gpu == 0: np.save(os.path.join(args.save_path, 'mis_advg.npy'), mis.data.cpu().numpy())

def accuracy(output, target, topk=(1,)):
    if len(target.shape) > 1: return torch.tensor(1), torch.tensor(1)

    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t().contiguous()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
    return res

if __name__ == '__main__': main()
