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

import models.resnet as models
from utils import AverageMeter, RecorderMeter, time_string, convert_secs2time, _ECELoss, load_dataset_in_ft, plot_mi, plot_ens, ent
from mean_field import *

model_names = sorted(name for name in models.__dict__ if name.islower() and not name.startswith("__") and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='Training script for ImageNet', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

# Data / Model
parser.add_argument('--data_path', metavar='DPATH', default='./data/ImageNet', type=str, help='Path to dataset')
parser.add_argument('--arch', metavar='ARCH', default='resnet50', help='model architecture: ' + ' | '.join(model_names) + ' (default: resnet50)')

# Optimization
parser.add_argument('--epochs', metavar='N', type=int, default=90, help='Number of epochs to train.')
parser.add_argument('--batch_size', type=int, default=256, help='Batch size.')
parser.add_argument('--learning_rate', type=float, default=0.1, help='The Learning Rate.')
parser.add_argument('--momentum', type=float, default=0.9, help='Momentum.')
parser.add_argument('--schedule', type=int, nargs='+', default=[1, 2, 3], help='Decrease learning rate at these epochs.')
parser.add_argument('--gammas', type=float, nargs='+', default=[0.1, 0.1, 0.1], help='LR is multiplied by gamma on schedule')

#Regularization
parser.add_argument('--decay', type=float, default=1e-4, help='Weight decay (L2 penalty).')
parser.add_argument('--dropout_rate', type=float, default=0.)

# Checkpoints
parser.add_argument('--save_path', type=str, default='./snapshots_ab_in/', help='Folder to save checkpoints and log.')
parser.add_argument('--resume', default='', type=str, metavar='PATH', help='Path to latest checkpoint (default: none)')
parser.add_argument('--start_epoch', default=0, type=int, metavar='N', help='Manual epoch number (useful on restarts)')
parser.add_argument('--evaluate', dest='evaluate', action='store_true', help='Evaluate model on test set')

# Acceleration
parser.add_argument('--workers', type=int, default=4, help='number of data loading workers (default: 2)')

# Random seed
parser.add_argument('--manualSeed', type=int, default=0, help='manual seed')
parser.add_argument('--job-id', type=str, default='')

# Bayesian
parser.add_argument('--bayes', type=str, default=None, help='Bayes type: None, mean field, matrix gaussian')
parser.add_argument('--log_sigma_init_range', type=float, nargs='+', default=[-6, -5])
parser.add_argument('--log_sigma_lr', type=float, default=0.1)
parser.add_argument('--single_eps', action='store_true', default=False)
parser.add_argument('--local_reparam', action='store_true', default=False)
parser.add_argument('--alpha', type=float, default=1.)
parser.add_argument('--max_choice', type=int, default=None)
parser.add_argument('--num_gan', type=int, default=100)
parser.add_argument('--aug_n', type=int, default=None)
parser.add_argument('--aug_m', type=int, default=None)
parser.add_argument('--mi_th', type=float, default=0.5)

# GAN generated data augmentation
parser.add_argument('--blur_prob', type=float, default=0.5)
parser.add_argument('--blur_sig', type=float, nargs='+', default=[0., 3.])
parser.add_argument('--jpg_prob', type=float, default=0.5)
parser.add_argument('--jpg_method', type=str, nargs='+', default=['cv2', 'pil'])
parser.add_argument('--jpg_qual', type=int, nargs='+', default=[30, 100])

# attack settings
parser.add_argument('--epsilon', default=16./255., type=float,
                    help='perturbation')
parser.add_argument('--epsilon_scale', default=1., type=float)
parser.add_argument('--num-steps', default=20, type=int,
                    help='perturb number of steps')
parser.add_argument('--step-size', default=1./255., type=float,
                    help='perturb step size')
parser.add_argument('--random',
                    default=True,
                    help='random initialization for PGD')

# Dist
parser.add_argument('--world-size', default=1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=0, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://127.0.0.1', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-port', default='1234', type=str,
                    help='port used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')

best_acc = 0

def main():
    args = parser.parse_args()
    if not os.path.isdir(args.data_path): os.makedirs(args.data_path)
    job_id = args.job_id
    args.save_path = args.save_path + job_id
    if not os.path.isdir(args.save_path): os.makedirs(args.save_path)

    args.use_cuda = torch.cuda.is_available()
    if args.manualSeed is None: args.manualSeed = random.randint(1, 10000)
    random.seed(args.manualSeed)
    np.random.seed(args.manualSeed)
    torch.manual_seed(args.manualSeed)
    if args.use_cuda: torch.cuda.manual_seed_all(args.manualSeed)
    cudnn.deterministic = True

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')
    else:
        args.multiprocessing_distributed = True

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed
    ngpus_per_node = torch.cuda.device_count()
    if args.multiprocessing_distributed:
        args.world_size = ngpus_per_node * args.world_size
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        main_worker(args.gpu, ngpus_per_node, args)

def main_worker(gpu, ngpus_per_node, args):
    global best_acc
    args.gpu = gpu
    assert args.gpu is not None
    print("Use GPU: {} for training".format(args.gpu))

    log = open(os.path.join(args.save_path, 'log{}{}.txt'.format('_seed'+
                   str(args.manualSeed), '_eval' if args.evaluate else '')), 'w')
    log = (log, args.gpu)

    net = models.__dict__[args.arch](args)
    print_log("Python version : {}".format(sys.version.replace('\n', ' ')), log)
    print_log("PyTorch  version : {}".format(torch.__version__), log)
    print_log("CuDNN  version : {}".format(torch.backends.cudnn.version()), log)
    print_log("Number of parameters: {}".format(sum([p.numel() for p in net.parameters()])), log)
    print_log(str(args), log)

    if args.distributed:
        if args.multiprocessing_distributed:
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url+":"+args.dist_port,
                                world_size=args.world_size, rank=args.rank)
        torch.cuda.set_device(args.gpu)
        net.cuda(args.gpu)
        args.batch_size = int(args.batch_size / ngpus_per_node)
        net = torch.nn.parallel.DistributedDataParallel(net, device_ids=[args.gpu])
    else:
        torch.cuda.set_device(args.gpu)
        net = net.cuda(args.gpu)

    criterion = torch.nn.CrossEntropyLoss().cuda(args.gpu)

    mus, vars = [], []
    for name, param in net.named_parameters():
        if 'log_sigma' in name: vars.append(param)
        else: assert(param.requires_grad); mus.append(param)
    optimizer = torch.optim.SGD(mus, args.learning_rate,
                                momentum=args.momentum,
                                weight_decay=args.decay)
    if args.bayes:
        assert(len(mus) == len(vars))
        var_optimizer = VarSGD(vars, args.log_sigma_lr, num_data=None,
                               momentum=args.momentum, weight_decay=args.decay)
    else:
        assert(len(vars) == 0)
        var_optimizer = NoneOptimizer()

    net_dict = net.state_dict()
    prefix_ = 'module.' if args.distributed else ''
    net_dict.update({prefix_ + k + '_mu'if args.bayes and ('weight' in k or 'bias' in k) else prefix_ + k: v for k,v in torch.load('./ckpts/resnet50-19c8e357.pth', map_location='cuda:{}'.format(args.gpu)).items()})
    net.load_state_dict(net_dict)

    recorder = RecorderMeter(args.epochs)
    if args.resume:
        if args.resume == 'auto': args.resume = os.path.join(args.save_path, 'checkpoint.pth.tar')
        if os.path.isfile(args.resume):
            print_log("=> loading checkpoint '{}'".format(args.resume), log)
            checkpoint = torch.load(args.resume, map_location='cuda:{}'.format(args.gpu))
            if 'recorder' in checkpoint:
                recorder = checkpoint['recorder']
                recorder.refresh(args.epochs)
            args.start_epoch = checkpoint['epoch']
            net.load_state_dict(checkpoint['state_dict'] if args.distributed else {k.replace('module.', ''): v for k,v in checkpoint['state_dict'].items()})
            optimizer.load_state_dict(checkpoint['optimizer'])
            var_optimizer.load_state_dict(checkpoint['var_optimizer'])
            best_acc = recorder.max_accuracy(False)
            print_log("=> loaded checkpoint '{}' accuracy={} (epoch {})" .format(args.resume, best_acc, checkpoint['epoch']), log)
        else:
            print_log("=> no checkpoint found at '{}'".format(args.resume), log)
    else:
        print_log("=> do not use any checkpoint for {} model".format(args.arch), log)

    cudnn.benchmark = True

    train_loader, train_loader1, test_loader, test_loader1, fake_loader, adv_loader = load_dataset_in_ft(args)
    var_optimizer.num_data = len(train_loader.dataset)

    if args.evaluate:
        evaluate(test_loader, test_loader1, fake_loader, adv_loader, net, criterion, args, log, 20, 100)
        return

    start_time = time.time()
    epoch_time = AverageMeter()
    train_los = -1

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_loader.sampler.set_epoch(epoch)
            train_loader1.sampler.set_epoch(epoch)
        cur_lr, cur_slr = adjust_learning_rate(optimizer, var_optimizer, epoch, args)

        need_hour, need_mins, need_secs = convert_secs2time(epoch_time.avg * (args.epochs-epoch))
        need_time = '[Need: {:02d}:{:02d}:{:02d}]'.format(need_hour, need_mins, need_secs)

        print_log('\n==>>{:s} [Epoch={:03d}/{:03d}] {:s} [learning_rate={:6.4f} {:6.4f}]'.format(
                                    time_string(), epoch, args.epochs, need_time, cur_lr, cur_slr) \
                    + ' [Best : Accuracy={:.2f}, Error={:.2f}]'.format(recorder.max_accuracy(False), 100-recorder.max_accuracy(False)), log)

        train_acc, train_los = train(train_loader, train_loader1, net, criterion, optimizer, var_optimizer, epoch, args, log)
        if epoch == 5:
            val_acc, val_los   = evaluate(test_loader, test_loader1, fake_loader, adv_loader, net, criterion, args, log, 2)
        else:
            val_acc, val_los = 0, 0
        recorder.update(epoch, train_los, train_acc, val_los, val_acc)

        is_best = False
        if val_acc > best_acc:
            is_best = True
            best_acc = val_acc

        if args.gpu == 0:
            save_checkpoint({
              'epoch': epoch + 1,
              'arch': args.arch,
              'state_dict': net.state_dict(),
              'recorder': recorder,
              'optimizer' : optimizer.state_dict(),
              'var_optimizer' : var_optimizer.state_dict(),
            }, False, args.save_path, 'checkpoint.pth.tar')

        epoch_time.update(time.time() - start_time)
        start_time = time.time()
        recorder.plot_curve(os.path.join(args.save_path, 'log.png'))

    evaluate(test_loader, test_loader1, fake_loader, adv_loader, net, criterion, args, log, 20, 100)
    log[0].close()

def train(train_loader, train_loader1, model, criterion, optimizer, var_optimizer, epoch, args, log):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    rk_losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    train_loader1_iter = iter(train_loader1)

    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        data_time.update(time.time() - end)

        input = input.cuda(args.gpu, non_blocking=True)
        target = target.cuda(args.gpu, non_blocking=True)


        input1 = next(train_loader1_iter)
        input1 = input1.cuda(args.gpu, non_blocking=True)

        bs = input.shape[0]
        bs1 = input1.shape[0]

        output = model(torch.cat([input, input1.repeat(2, 1, 1, 1)]))
        loss = criterion(output[:bs], target)

        out1_0 = output[bs:bs+bs1].softmax(-1)
        out1_1 = output[bs+bs1:].softmax(-1)
        mi1 = ent((out1_0 + out1_1)/2.) - (ent(out1_0) + ent(out1_1))/2.
        rank_loss = torch.nn.functional.relu(args.mi_th - mi1).mean()

        prec1, prec5 = accuracy(output[:bs], target, topk=(1, 5))
        losses.update(loss.detach().item(), bs)
        rk_losses.update(rank_loss.detach().item(), bs1)
        top1.update(prec1.item(), bs)
        top5.update(prec5.item(), bs)

        optimizer.zero_grad()
        var_optimizer.zero_grad()
        (loss+rank_loss*args.alpha).backward()
        optimizer.step()
        var_optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()

        if i == len(train_loader) - 1:
            print_log('  Epoch: [{:03d}][{:03d}/{:03d}]   '
                        'Time {batch_time.avg:.3f}   '
                        'Data {data_time.avg:.3f}   '
                        'Loss {loss.avg:.4f}   '
                        'RK Loss {rk_loss.avg:.4f}   '
                        'Prec@1 {top1.avg:.3f}   '
                        'Prec@5 {top5.avg:.3f}   '.format(
                        epoch, i, len(train_loader), batch_time=batch_time, rk_loss=rk_losses,
                        data_time=data_time, loss=losses, top1=top1, top5=top5) + time_string(), log)
    return top1.avg, losses.avg

def evaluate(test_loader, test_loader1, fake_loader, adv_loader, net, criterion, args, log, nums=100, nums2=None):
    if args.bayes: net.apply(freeze)
    deter_rets = ens_validate(test_loader, net, criterion, args, log, False, 1)
    if args.bayes: net.apply(unfreeze)
    if not args.bayes and args.dropout_rate == 0: nums = 1; nums2=1
    if not nums2: nums2 = nums

    rets = ens_validate(test_loader, net, criterion, args, log, True, nums2)
    print_log('TOP1 average: {:.4f}, ensemble: {:.4f}, deter: {:.4f}'.format(rets[:,2].mean(), rets[-1][-3], deter_rets[0][2]), log)
    print_log('TOP5 average: {:.4f}, ensemble: {:.4f}, deter: {:.4f}'.format(rets[:,3].mean(), rets[-1][-2], deter_rets[0][3]), log)
    print_log('LOS  average: {:.4f}, ensemble: {:.4f}, deter: {:.4f}'.format(rets[:,1].mean(), rets[-1][-4], deter_rets[0][1]), log)
    print_log('ECE  ensemble: {:.4f}, deter: {:.4f}'.format(rets[-1][-1], deter_rets[-1][-1]), log)
    if args.gpu == 0: plot_ens(args.save_path, rets, deter_rets[0][2])

    ens_attack(test_loader1, net, criterion, args, log, nums)
    if args.gpu == 0: print_log('NAT vs. ADV: AP {}'.format(plot_mi(args.save_path, 'advg')), log)

    ens_validate(adv_loader, net, criterion, args, log, True, nums, suffix='_adv')
    if args.gpu == 0: print_log('NAT vs. ADV-dyp: AP {}'.format(plot_mi(args.save_path, 'adv')), log)

    ens_validate(fake_loader, net, criterion, args, log, True, nums, suffix='_fake')
    if args.gpu == 0: print_log('NAT vs. Fake: AP {}'.format(plot_mi(args.save_path, 'fake')), log)
    return rets[-1][-3], rets[-1][-4]

def ens_validate(val_loader, model, criterion, args, log, unfreeze_dropout=False, num_ens=100, suffix=''):
    model.eval()
    if unfreeze_dropout and args.dropout_rate > 0.:
        for m in model.modules():
            if m.__class__.__name__.startswith('Dropout'): m.train()

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
                output = model(input)

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

        # to sync
        confidences, predictions = torch.max(preds, 1)
        targets = torch.cat(targets, 0)
        mis = (- preds * preds.log()).sum(1) - (0 if num_ens == 1 else torch.cat(mis, 0))
        rets /= targets.size(0)

        if args.distributed:
            if suffix == '':
                confidences = dist_collect(confidences)
                predictions = dist_collect(predictions)
                targets = dist_collect(targets)
            mis = dist_collect(mis)
            rets = reduce_tensor(rets.data, args)
        rets = rets.data.cpu().numpy()
        if suffix == '':
            ens_ece = ece_func(confidences, predictions, targets, os.path.join(args.save_path, 'ens_cal{}.pdf'.format(suffix)))
            rets[-1, -1] = ens_ece

    if args.gpu == 0:
        np.save(os.path.join(args.save_path, 'mis{}.npy'.format(suffix)), mis.data.cpu().numpy())
    return rets

def ens_attack(val_loader, model, criterion, args, log, num_ens=100, joint=False):
    def _grad(X, y, mean, std):
        if joint:
            with model.no_sync():
                with torch.enable_grad():
                    X.requires_grad_()
                    outputs = []
                    for j in range(num_ens):
                        output = model(X.sub(mean).div(std))
                        outputs.append(output)
                    outputs = torch.stack(outputs)
                    preds = outputs.softmax(-1).mean(0)
                    mis = (-preds * (preds+1e-8).log()).sum(1) + (outputs.softmax(-1)*outputs.log_softmax(-1)).sum(2).mean(0)
                    loss = torch.nn.functional.nll_loss((preds+1e-8).log(), y, reduction='none')-mis
                    grad_ = torch.autograd.grad(
                        [loss], [X], grad_outputs=torch.ones_like(loss), retain_graph=False)[0].detach()
        else:
            probs = torch.zeros(num_ens, X.shape[0]).cuda(args.gpu)
            grads = torch.zeros(num_ens, *list(X.shape)).cuda(args.gpu)
            for j in range(num_ens):
                with model.no_sync():
                    with torch.enable_grad():
                        X.requires_grad_()
                        output = model(X.sub(mean).div(std))
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
            output = model(X_pgd.sub(mean).div(std))
            mis = (mis * ens + (-output.softmax(-1) * (output).log_softmax(-1)).sum(1)) / (ens + 1)
            preds = (preds * ens + output.softmax(-1)) / (ens + 1)

        loss = criterion((preds+1e-8).log(), target)
        prec1, prec5 = accuracy(preds, target, topk=(1, 5))
        mis = (- preds * (preds+1e-8).log()).sum(1) - (0 if num_ens == 1 else mis)
        return loss, prec1, prec5, mis

    mean = torch.from_numpy(np.array([0.485, 0.456, 0.406])).view(1,3,1,1).cuda(args.gpu).float()
    std = torch.from_numpy(np.array([0.229, 0.224, 0.225])).view(1,3,1,1).cuda(args.gpu).float()

    losses, top1, top5 = 0, 0, 0
    model.eval()
    if args.dropout_rate > 0.:
        for m in model.modules():
            if m.__class__.__name__.startswith('Dropout'): m.train()
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
        losses = reduce_tensor(losses.data, args)
        top1 = reduce_tensor(top1.data, args)
        top5 = reduce_tensor(top5.data, args)

        if args.distributed: mis = dist_collect(mis)

    print_log('ADV ensemble TOP1: {:.4f}, TOP5: {:.4f}, LOS: {:.4f}'.format(top1.item(), top5.item(), losses.item()), log)
    if args.gpu == 0: np.save(os.path.join(args.save_path, 'mis_advg.npy'), mis.data.cpu().numpy())

def print_log(print_string, log):
    if log[1] == 0:
        print("{}".format(print_string))
        log[0].write('{}\n'.format(print_string))
        log[0].flush()

def save_checkpoint(state, is_best, save_path, filename):
    filename = os.path.join(save_path, filename)
    torch.save(state, filename)
    if is_best:
        bestname = os.path.join(save_path, 'model_best.pth.tar')
        shutil.copyfile(filename, bestname)

def adjust_learning_rate(optimizer, var_optimizer, epoch, args):
    lr = args.learning_rate
    slr = args.log_sigma_lr
    assert len(args.gammas) == len(args.schedule), "length of gammas and schedule should be equal"
    for (gamma, step) in zip(args.gammas, args.schedule):
        if (epoch >= step): slr = slr * gamma
        else: break
    lr = lr * np.prod(args.gammas)
    for param_group in optimizer.param_groups: param_group['lr'] = lr
    for param_group in var_optimizer.param_groups: param_group['lr'] = slr
    return lr, slr

def accuracy(output, target, topk=(1,)):
    if len(target.shape) > 1: return torch.tensor(1), torch.tensor(1)

    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
    return res

def reduce_tensor(tensor, args):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= args.world_size
    return rt

def dist_collect(x):
    """ collect all tensor from all GPUs
    args:
        x: shape (mini_batch, ...)
    returns:
        shape (mini_batch * num_gpu, ...)
    """
    x = x.contiguous()
    out_list = [torch.zeros_like(x, device=x.device, dtype=x.dtype)
                for _ in range(dist.get_world_size())]
    dist.all_gather(out_list, x)
    return torch.cat(out_list, dim=0)

def freeze(m):
    if isinstance(m, (BayesConv2dMF, BayesLinearMF, BayesBatchNorm2dMF)):
        m.deterministic = True

def unfreeze(m):
    if isinstance(m, (BayesConv2dMF, BayesLinearMF, BayesBatchNorm2dMF)):
        m.deterministic = False

if __name__ == '__main__': main()
