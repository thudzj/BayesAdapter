import argparse
import os
import shutil
import time
import random
import warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets

import numpy as np

try:
    from apex.parallel import DistributedDataParallel as DDP
    from apex.fp16_utils import *
    from apex import amp, optimizers
    from apex.multi_tensor_apply import multi_tensor_applier
except ImportError:
    raise ImportError("Please install apex from https://www.github.com/nvidia/apex to run this example.")

import models.resnet as models
from utils import _ECELoss, plot_mi, plot_ens, convert_secs2time, time_string, fast_collate, data_prefetcher
from mean_field import *

def parse():
    model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

    parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
    parser.add_argument('--data', metavar='DIR', default='/data/LargeData/ImageNet',
                        help='path to dataset')
    parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet50',
                        choices=model_names,
                        help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet50)')
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--epochs', default=90, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('-b', '--batch-size', default=32, type=int,
                        metavar='N', help='mini-batch size per process (default: 256)')
    parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                        metavar='LR', help='Initial learning rate.  Will be scaled by <global batch size>/256: args.lr = args.lr*float(args.batch_size*args.world_size)/256.  A warmup schedule will also be applied over the first 5 epochs.')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                        help='evaluate model on validation set')

    parser.add_argument("--local_rank", default=0, type=int)
    parser.add_argument('--sync_bn', action='store_true',
                        help='enabling apex sync BN.')

    parser.add_argument('--opt-level', type=str)
    parser.add_argument('--keep-batchnorm-fp32', type=str, default=None)
    parser.add_argument('--loss-scale', type=str, default=None)

    # Bayesian
    parser.add_argument('--dropout_rate', type=float, default=0.)
    parser.add_argument('--bayes', type=str, default=None, help='Bayes type: None, mean field, matrix gaussian')
    parser.add_argument('--log_sigma_init_range', type=float, nargs='+', default=[-5, -4])
    parser.add_argument('--log_sigma_lr', type=float, default=0.1)
    parser.add_argument('--single_eps', action='store_true', default=False)
    parser.add_argument('--local_reparam', action='store_true', default=False)

    # ckpt
    parser.add_argument('--save_path', type=str, default='/data/zhijie/snapshots_ab_in/', help='Folder to save checkpoints and log.')
    parser.add_argument('--job-id', type=str, default='')
    parser.add_argument('--manualSeed', type=int, default=0, help='manual seed')

    args = parser.parse_args()
    return args

def main():
    global best_prec1, args

    args = parse()
    args.save_path = args.save_path + args.job_id
    if not os.path.isdir(args.save_path): os.makedirs(args.save_path)

    log = open(os.path.join(args.save_path, 'log{}{}.txt'.format('_seed'+
               str(args.manualSeed), '_eval' if args.evaluate else '')), 'w') if args.local_rank == 0 else None
    log = (log, args.local_rank)

    print_log("CUDNN VERSION: {}".format(torch.backends.cudnn.version()), log)
    print_log(str(args), log)

    if args.manualSeed is None: args.manualSeed = random.randint(1, 10000)
    random.seed(args.manualSeed)
    np.random.seed(args.manualSeed)
    torch.manual_seed(args.manualSeed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(args.manualSeed)
    cudnn.deterministic = True

    best_prec1 = 0

    args.distributed = False
    if 'WORLD_SIZE' in os.environ:
        args.distributed = int(os.environ['WORLD_SIZE']) > 1

    args.gpu = 0
    args.world_size = 1

    if args.distributed:
        args.gpu = args.local_rank
        torch.cuda.set_device(args.gpu)
        torch.distributed.init_process_group(backend='nccl',
                                             init_method='env://')
        args.world_size = torch.distributed.get_world_size()

    assert torch.backends.cudnn.enabled, "Amp requires cudnn backend to be enabled."

    memory_format = torch.contiguous_format

    # create model
    print_log("=> creating model '{}'".format(args.arch), log)
    model = models.__dict__[args.arch](args)
    print_log("Number of parameters: {}".format(sum([p.numel() for p in model.parameters()])), log)

    if args.sync_bn:
        import apex
        print_log("using apex synced BN", log)
        model = apex.parallel.convert_syncbn_model(model)

    model = model.cuda().to()

    # Scale learning rate based on global batch size
    args.lr = args.lr*float(args.batch_size*args.world_size)/256.
    args.log_sigma_lr = args.log_sigma_lr*float(args.batch_size*args.world_size)/256.
    mus, vars = [], []
    for name, param in model.named_parameters():
        if 'log_sigma' in name: vars.append(param)
        else: assert(param.requires_grad); mus.append(param)
    optimizer = torch.optim.SGD(mus, args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
    if args.bayes:
        assert(len(mus) == len(vars))
        var_optimizer = VarSGD(vars, args.log_sigma_lr, num_data=None,
                               momentum=args.momentum, weight_decay=args.weight_decay)
    else:
        assert(len(vars) == 0)
        var_optimizer = None

    # Initialize Amp.  Amp accepts either values or strings for the optional override arguments,
    # for convenient interoperation with argparse.
    if args.bayes:
        model, [optimizer, var_optimizer] = amp.initialize(model, [optimizer, var_optimizer],
                                          opt_level=args.opt_level,
                                          keep_batchnorm_fp32=args.keep_batchnorm_fp32,
                                          loss_scale=args.loss_scale
                                          )
    else:
        model, optimizer = amp.initialize(model, optimizer,
                                          opt_level=args.opt_level,
                                          keep_batchnorm_fp32=args.keep_batchnorm_fp32,
                                          loss_scale=args.loss_scale
                                          )

    # For distributed training, wrap the model with apex.parallel.DistributedDataParallel.
    # This must be done AFTER the call to amp.initialize.  If model = DDP(model) is called
    # before model, ... = amp.initialize(model, ...), the call to amp.initialize may alter
    # the types of model's parameters in a way that disrupts or destroys DDP's allreduce hooks.
    if args.distributed:
        # By default, apex.parallel.DistributedDataParallel overlaps communication with
        # computation in the backward pass.
        # model = DDP(model)
        # delay_allreduce delays all communication to the end of the backward pass.
        model = DDP(model, delay_allreduce=True)

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda()

    # Optionally resume from a checkpoint
    if args.resume:
        # Use a local scope to avoid dangling references
        def resume():
            if os.path.isfile(args.resume):
                print_log("=> loading checkpoint '{}'".format(args.resume), log)
                checkpoint = torch.load(args.resume, map_location = lambda storage, loc: storage.cuda(args.gpu))
                args.start_epoch = checkpoint['epoch']
                best_prec1 = checkpoint['best_prec1']
                model.load_state_dict(checkpoint['state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer'])
                if args.bayes:
                    var_optimizer.load_state_dict(checkpoint['var_optimizer'])
                if 'amp' in checkpoint:
                    amp.load_state_dict(checkpoint['amp'])
                print_log("=> loaded checkpoint '{}' (epoch {})"
                      .format(args.resume, checkpoint['epoch']), log)
            else:
                print_log("=> no checkpoint found at '{}'".format(args.resume), log)
        resume()

    cudnn.benchmark = True

    # Data loading code
    traindir = os.path.join(args.data, 'train')
    valdir = os.path.join(args.data, 'val')

    if(args.arch == "inception_v3"):
        raise RuntimeError("Currently, inception_v3 is not supported by this example.")
        # crop_size = 299
        # val_size = 320 # I chose this value arbitrarily, we can adjust.
    else:
        crop_size = 224
        val_size = 256

    train_dataset = datasets.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.RandomResizedCrop(crop_size),
            transforms.RandomHorizontalFlip(),
            # transforms.ToTensor(), Too slow
            # normalize,
        ]))
    val_dataset = datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Resize(val_size),
            transforms.CenterCrop(crop_size),
        ]))

    train_sampler = None
    val_sampler = None
    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset)

    collate_fn = lambda b: fast_collate(b, memory_format)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler, collate_fn=collate_fn)

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True,
        sampler=val_sampler,
        collate_fn=collate_fn)

    if args.bayes: var_optimizer.num_data = len(train_loader.dataset)

    if args.evaluate:
        validate(val_loader, model, criterion, log)
        return

    start_time = time.time()
    epoch_time = AverageMeter()
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)

        need_hour, need_mins, need_secs = convert_secs2time(epoch_time.avg * (args.epochs-epoch))
        need_time = '[Need: {:02d}:{:02d}:{:02d}]'.format(need_hour, need_mins, need_secs)

        print_log('\n==>>{:s} [Epoch={:03d}/{:03d}] {:s} [learning_rate={:6.4f}]'.format(time_string(), epoch, args.epochs, need_time, optimizer.param_groups[0]['lr']) \
                    + ' [Best : Accuracy={:.2f}, Error={:.2f}]'.format(best_prec1, 100-best_prec1), log)

        # train for one epoch
        train(train_loader, model, criterion, optimizer, var_optimizer, epoch, log)

        # evaluate on validation set
        prec1 = validate(val_loader, model, criterion, log)

        # remember best prec@1 and save checkpoint
        if args.local_rank == 0:
            is_best = prec1 > best_prec1
            best_prec1 = max(prec1, best_prec1)
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'best_prec1': best_prec1,
                'optimizer' : optimizer.state_dict(),
                'var_optimizer' : None if not args.bayes else var_optimizer.state_dict(),
                'amp': amp.state_dict()
            }, is_best, args.save_path)
        epoch_time.update(time.time() - start_time)
        start_time = time.time()
    if args.local_rank == 0:
        log[0].close()

def train(train_loader, model, criterion, optimizer, var_optimizer, epoch, log):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode
    model.train()
    end = time.time()

    prefetcher = data_prefetcher(train_loader)
    input, target = prefetcher.next()
    i = 0
    while input is not None:
        i += 1
        adjust_learning_rate(optimizer, var_optimizer, epoch, i, len(train_loader))

        # compute output
        output = model(input)
        loss = criterion(output, target)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        if args.bayes: var_optimizer.zero_grad()

        if args.bayes:
            with amp.scale_loss(loss, [optimizer, var_optimizer]) as scaled_loss:
                scaled_loss.backward()
        else:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()

        optimizer.step()
        if args.bayes: var_optimizer.step()

        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        losses.update(to_python_float(loss.data), input.size(0))
        top1.update(to_python_float(prec1), input.size(0))
        top5.update(to_python_float(prec5), input.size(0))
        batch_time.update((time.time() - end))
        end = time.time()

        input, target = prefetcher.next()

    print_log('Epoch: [{0}]\t'
          'Time {batch_time.avg:.3f}\t'
          'Speed {1:.3f}\t'
          'Loss {loss.avg:.4f}\t'
          'Prec@1 {top1.avg:.3f}\t'
          'Prec@5 {top5.avg:.3f}'.format(
           epoch, args.world_size*args.batch_size/batch_time.avg,
           batch_time=batch_time, loss=losses, top1=top1, top5=top5), log)


def validate(val_loader, model, criterion, log):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()

    prefetcher = data_prefetcher(val_loader)
    input, target = prefetcher.next()
    i = 0
    while input is not None:
        i += 1

        # compute output
        with torch.no_grad():
            output = model(input)
            loss = criterion(output, target)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))

        if args.distributed:
            reduced_loss = reduce_tensor(loss.data)
            prec1 = reduce_tensor(prec1)
            prec5 = reduce_tensor(prec5)
        else:
            reduced_loss = loss.data

        losses.update(to_python_float(reduced_loss), input.size(0))
        top1.update(to_python_float(prec1), input.size(0))
        top5.update(to_python_float(prec5), input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        input, target = prefetcher.next()

    print_log(' ** Test: \t'
          'Time {batch_time.avg:.3f}\t'
          'Speed {0:.3f}\t'
          'Loss {loss.avg:.4f}\t'
          'Prec@1 {top1.avg:.3f}\t'
          'Prec@5 {top5.avg:.3f}'.format(
           args.world_size * args.batch_size / batch_time.avg,
           batch_time=batch_time, loss=losses,
           top1=top1, top5=top5), log)

    return top1.avg

def save_checkpoint(state, is_best, dir_, filename='checkpoint.pth.tar'):
    torch.save(state, os.path.join(dir_, filename))
    if is_best:
        shutil.copyfile(os.path.join(dir_, filename), os.path.join(dir_, 'model_best.pth.tar'))


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(optimizer, var_optimizer, epoch, step, len_epoch):
    """LR schedule that should yield 76% converged accuracy with batch size 256"""
    factor = epoch // 30

    if epoch >= 80:
        factor = factor + 1

    lr = args.lr*(0.1**factor)
    slr = args.log_sigma_lr*(0.1**factor)

    """Warmup"""
    if epoch < 5:
        lr = lr*float(1 + step + epoch*len_epoch)/(5.*len_epoch)
        slr = slr*float(1 + step + epoch*len_epoch)/(5.*len_epoch)

    if epoch < 5 or (epoch >= 5 and step == 1):
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        if args.bayes:
            for param_group in var_optimizer.param_groups:
                param_group['lr'] = slr

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
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

def reduce_tensor(tensor):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= args.world_size
    return rt

def print_log(print_string, log):
    if log[1] == 0:
        print("{}".format(print_string))
        log[0].write('{}\n'.format(print_string))
        log[0].flush()

if __name__ == '__main__':
    main()
