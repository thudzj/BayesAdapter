from __future__ import division
import os, sys, shutil, time, random, math
import argparse
import warnings
import numpy as np
from contextlib import suppress
warnings.filterwarnings("ignore")

import torch
import torch.backends.cudnn as cudnn

import torch.nn.parallel
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.utils.data.distributed

import models.mobilenet as models
from utils import AverageMeter, RecorderMeter, time_string, convert_secs2time, load_dataset_face_ft, plot_mi, ent
from verification import verify

sys.path.insert(0, '/data/zhijie/ScalableBDL/')
from scalablebdl.low_rank import to_bayesian as to_bayesian_lr
from scalablebdl.mean_field import to_bayesian as to_bayesian_mf
from scalablebdl.bnn_utils import freeze, unfreeze, set_mc_sample_id
from scalablebdl.prior_reg import PriorRegularizor

parser = argparse.ArgumentParser(description='Bayesian fine-tuning script',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

# Data / Model
parser.add_argument('--data_path', metavar='DPATH', default='/data/xiaoyang/data/faces_emore/', type=str, help='Path to dataset')
parser.add_argument('--arch', metavar='ARCH', default='mobilenet_v2', help='model architecture')

# Optimization
parser.add_argument('--epochs', metavar='N', type=int, default=90, help='Number of epochs to train.')
parser.add_argument('--batch_size', type=int, default=256, help='Batch size.')
parser.add_argument('--ft_lr', type=float, default=1e-4, help='The Learning Rate.')
parser.add_argument('--momentum', type=float, default=0.9, help='Momentum.')
parser.add_argument('--schedule', type=int, nargs='+', default=[1, 2, 3], help='Decrease learning rate at these epochs.')
parser.add_argument('--gammas', type=float, nargs='+', default=[0.1, 0.1, 0.1], help='LR is multiplied by gamma on schedule')

#Regularization
parser.add_argument('--decay', type=float, default=1e-4, help='Weight decay (L2 penalty).')
parser.add_argument('--dropout_rate', type=float, default=0.)

# Checkpoints
parser.add_argument('--save_path', type=str, default='/data/zhijie/snapshots_ab_face/', help='Folder to save checkpoints and log.')
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
parser.add_argument('--psi_init_range', type=float, nargs='+', default=[-6, -5])
parser.add_argument('--pert_init_std', type=float, default=0.2)
parser.add_argument('--lr', type=float, default=0.1)
parser.add_argument('--lr_min', type=float, default=None)
parser.add_argument('--num_mc_samples', type=int, default=20)
parser.add_argument('--low_rank', type=int, default=1)
parser.add_argument('--alpha', type=float, default=1.)
parser.add_argument('--max_choice', type=int, default=None)
parser.add_argument('--num_gan', type=int, default=100)
parser.add_argument('--aug_n', type=int, default=None)
parser.add_argument('--aug_m', type=int, default=None)
parser.add_argument('--mi_th', type=float, default=0.5)
parser.add_argument('--cos_lr', action='store_true', default=False)
# for the old version mean_field
# parser.add_argument('--single_eps', action='store_true', default=False)
# parser.add_argument('--local_reparam', action='store_true', default=False)

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

# Speedup
parser.add_argument('--amp', action='store_true', default=False,
                    help='use Native AMP for mixed precision training')

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

    net = models.__dict__[args.arch](args, num_classes=10341)
    net.load_state_dict({k.replace('module.', ''): v for k, v in
        torch.load('./ckpts/mobilenet_v2-face-ft0.01-drop0-decay5.pth', map_location='cpu')['state_dict'].items()})

    if args.bayes == 'mf':
        net = to_bayesian_mf(net, args.psi_init_range, args.num_mc_samples)
    elif args.bayes == 'low_rank':
        net = to_bayesian_lr(net, args.low_rank, args.num_mc_samples, args.pert_init_std)
    else:
        assert args.bayes is None
        args.num_mc_samples = 1

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

    pre_trained, new_added = [], []
    for name, param in net.named_parameters():
        if (args.bayes == 'mf' and 'psi' in name) \
                or (args.bayes == 'low_rank' and ('perturbations' in name)):
            new_added.append(param)
        else:
            pre_trained.append(param)
    args.lr_min = args.ft_lr
    optimizer = torch.optim.SGD([{'params': pre_trained, 'lr': args.ft_lr},
                                 {'params': new_added, 'lr': args.lr}],
                                 weight_decay=0, momentum=args.momentum)
    prior_regularizor = PriorRegularizor(net, args.decay, 450233,
                                         args.num_mc_samples, args.bayes)
    if args.amp:
        amp_autocast = torch.cuda.amp.autocast
        loss_scaler = torch.cuda.amp.GradScaler()
        print_log('Using native Torch AMP. Training in mixed precision.', log)
    else:
        amp_autocast = suppress  # do nothing
        loss_scaler = None
        print_log('AMP not enabled. Training in float32.', log)

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
            if not args.evaluate:
                optimizer.load_state_dict(checkpoint['optimizer'])
            if args.amp:
                loss_scaler.load_state_dict(checkpoint['loss_scaler'])
            best_acc = recorder.max_accuracy(False)
            print_log("=> loaded checkpoint '{}' accuracy={} (epoch {})" .format(
                args.resume, best_acc, checkpoint['epoch']), log)
        else:
            print_log("=> no checkpoint found at '{}'".format(args.resume), log)
    else:
        print_log("=> do not use any checkpoint for {} model".format(args.arch), log)

    cudnn.benchmark = True

    train_loader, train_loader1, test_loaders, fake_loader = load_dataset_face_ft(args)

    if args.evaluate:
        while True:
            evaluate(test_loaders, fake_loader, net, criterion, args, log, args.num_mc_samples, amp_autocast=amp_autocast)
        return

    start_time = time.time()
    epoch_time = AverageMeter()
    train_los = -1
    cur_lr, cur_slr = args.ft_lr, args.lr

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_loader.sampler.set_epoch(epoch)
            train_loader1.sampler.set_epoch(epoch)
        if not args.cos_lr:
            cur_lr, cur_slr = adjust_learning_rate(optimizer, epoch, args)

        need_hour, need_mins, need_secs = convert_secs2time(epoch_time.avg * (args.epochs-epoch))
        need_time = '[Need: {:02d}:{:02d}:{:02d}]'.format(need_hour, need_mins, need_secs)

        print_log('\n==>>{:s} [Epoch={:03d}/{:03d}] {:s} [learning_rate={:6.4f} {:6.4f}]'.format(
                                    time_string(), epoch, args.epochs, need_time, cur_lr, cur_slr) \
                    + ' [Best : Accuracy={:.2f}, Error={:.2f}]'.format(recorder.max_accuracy(False), 100-recorder.max_accuracy(False)), log)

        train_acc, train_los, cur_lr, cur_slr = train(train_loader, train_loader1,
            net, criterion, optimizer, epoch, args, log,
            prior_regularizor=prior_regularizor,
            amp_autocast=amp_autocast,
            loss_scaler=loss_scaler)

        recorder.update(epoch, train_los, train_acc, 0, 0)

        if args.gpu == 0:
            save_checkpoint({
              'epoch': epoch + 1,
              'arch': args.arch,
              'state_dict': net.state_dict(),
              'recorder': recorder,
              'optimizer' : optimizer.state_dict(),
              'loss_scaler' : loss_scaler.state_dict() if args.amp else None,
            }, False, args.save_path, 'checkpoint.pth.tar')

        epoch_time.update(time.time() - start_time)
        start_time = time.time()
        recorder.plot_curve(os.path.join(args.save_path, 'log.png'))

    while True:
        evaluate(test_loaders, fake_loader, net, criterion, args, log, args.num_mc_samples, amp_autocast=amp_autocast)

    log[0].close()

def train(train_loader, train_loader1, model, criterion, optimizer, epoch, args,
          log, prior_regularizor, amp_autocast=suppress, loss_scaler=None):
    cur_lr, cur_slr = None, None
    if args.alpha > 0:
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        unc_losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()

        train_loader1_iter = iter(train_loader1)

        model.train()

        end = time.time()
        for i, (input, target) in enumerate(train_loader):
            data_time.update(time.time() - end)

            if args.cos_lr:
                cur_lr, cur_slr = adjust_learning_rate_per_step(
                    optimizer, epoch, i, len(train_loader), args)

            input = input.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)

            input1 = next(train_loader1_iter)
            input1 = input1.cuda(args.gpu, non_blocking=True)

            bs = input.shape[0]
            bs1 = input1.shape[0]

            if args.bayes == 'low_rank':
                mc_sample_id = np.concatenate([np.random.randint(0, args.num_mc_samples, size=bs),
                    np.stack([np.random.choice(args.num_mc_samples, 2, replace=False) for _ in range(bs1)]).T.reshape(-1)])
                set_mc_sample_id(model, args.num_mc_samples, mc_sample_id)

            with amp_autocast():
                output = model(torch.cat([input, input1.repeat(2, 1, 1, 1)]))
                loss = criterion(output[:bs], target)

                out1_0 = output[bs:bs+bs1].softmax(-1)
                out1_1 = output[bs+bs1:].softmax(-1)
                mi1 = ent((out1_0 + out1_1)/2.) - (ent(out1_0) + ent(out1_1))/2.
                unc_loss = torch.nn.functional.relu(args.mi_th - mi1).mean()

            prec1, prec5 = accuracy(output[:bs], target, topk=(1, 5))
            losses.update(loss.detach().item(), bs)
            unc_losses.update(unc_loss.detach().item(), bs1)
            top1.update(prec1.item(), bs)
            top5.update(prec5.item(), bs)

            optimizer.zero_grad()
            if loss_scaler is not None:
                loss_scaler.scale(loss+unc_loss*args.alpha).backward()
                loss_scaler.unscale_(optimizer)
                prior_regularizor.step()
                loss_scaler.step(optimizer)
                loss_scaler.update()
            else:
                (loss+unc_loss*args.alpha).backward()
                prior_regularizor.step()
                optimizer.step()

            batch_time.update(time.time() - end)
            end = time.time()

            if i == len(train_loader) - 1:
                print_log('  Epoch: [{:03d}][{:03d}/{:03d}]   '
                            'Time {batch_time.avg:.3f}   '
                            'Data {data_time.avg:.3f}   '
                            'Loss {loss.avg:.4f}   '
                            'Unc Loss {unc_loss.avg:.4f}   '
                            'Prec@1 {top1.avg:.3f}   '
                            'Prec@5 {top5.avg:.3f}   '.format(
                            epoch, i, len(train_loader), batch_time=batch_time, unc_loss=unc_losses,
                            data_time=data_time, loss=losses, top1=top1, top5=top5) + time_string(), log)
    else:
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()

        model.train()

        end = time.time()
        for i, (input, target) in enumerate(train_loader):
            data_time.update(time.time() - end)

            if args.cos_lr:
                cur_lr, cur_slr = adjust_learning_rate_per_step(
                    optimizer, epoch, i, len(train_loader), args)

            input = input.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)
            if args.bayes == 'low_rank':
                mc_sample_id = np.random.randint(0, args.num_mc_samples, size=len(input))
                set_mc_sample_id(model, args.num_mc_samples, mc_sample_id)

            with amp_autocast():
                output = model(input)
                loss = criterion(output, target)

            prec1, prec5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.detach().item(), input.size(0))
            top1.update(prec1.item(), input.size(0))
            top5.update(prec5.item(), input.size(0))

            optimizer.zero_grad()
            if loss_scaler is not None:
                loss_scaler.scale(loss).backward()
                loss_scaler.unscale_(optimizer)
                prior_regularizor.step()
                loss_scaler.step(optimizer)
                loss_scaler.update()
            else:
                loss.backward()
                prior_regularizor.step()
                optimizer.step()

            batch_time.update(time.time() - end)
            end = time.time()

            if i == len(train_loader) - 1:
                print_log('  Epoch: [{:03d}][{:03d}/{:03d}]   '
                            'Time {batch_time.avg:.3f}   '
                            'Data {data_time.avg:.3f}   '
                            'Loss {loss.avg:.4f}   '
                            'Prec@1 {top1.avg:.3f}   '
                            'Prec@5 {top5.avg:.3f}   '.format(
                            epoch, i, len(train_loader), batch_time=batch_time,
                            data_time=data_time, loss=losses, top1=top1, top5=top5) + time_string(), log)
    return top1.avg, losses.avg, cur_lr, cur_slr

def evaluate(test_loaders, fake_loader, net, criterion, args, log, nums, amp_autocast=suppress):
    if args.bayes: net.apply(freeze)
    if args.gpu == 0: print("---------------------------deterministic---------------------------")
    ens_validate(test_loaders, net, criterion, args, log, False, 1, amp_autocast=amp_autocast)
    if args.bayes: net.apply(unfreeze)
    if not args.bayes and args.dropout_rate == 0: nums = 1

    if args.gpu == 0: print("---------------------------ensemble {} times---------------------------".format(nums))
    ens_validate(test_loaders, net, criterion, args, log, True, nums, amp_autocast=amp_autocast)

    ens_attack(test_loaders, net, criterion, args, log, nums, amp_autocast=amp_autocast)
    if args.gpu == 0:
        for k in test_loaders:
            print_log('{} vs. adversarial: AP {}'.format(k[0], plot_mi(args.save_path, 'adv_'+k[0], k[0])), log)
    ens_validate(fake_loader, net, criterion, args, log, True, nums, suffix='fake', amp_autocast=amp_autocast)
    if args.gpu == 0:
        for k in test_loaders:
            print_log('{} vs. DeepFake: AP {}'.format(k[0], plot_mi(args.save_path, 'fake', k[0])), log)

def ens_validate(val_loaders, model, criterion, args, log, unfreeze_dropout=False,
                 num_ens=100, suffix='', amp_autocast=suppress):
    model.eval()
    if unfreeze_dropout and args.dropout_rate > 0.:
        for m in model.modules():
            if m.__class__.__name__.startswith('Dropout'): m.train()

    if isinstance(val_loaders, list):
        name, val_loader, issame = val_loaders[args.gpu % len(val_loaders)]
    else:
        name, val_loader, issame = suffix, val_loaders, None
    with torch.no_grad():
        with model.no_sync():
            embeddings = []
            mis = [0 for _ in range(len(val_loader))]
            preds = [0 for _ in range(len(val_loader))]
            for i, input in enumerate(val_loader):
                if isinstance(input, tuple) or isinstance(input, list): input = input[0]
                input = input.cuda(args.gpu, non_blocking=True)

                with amp_autocast():
                    embedding_b = 0
                    for ens in range(num_ens):
                        if args.bayes == 'low_rank':
                            set_mc_sample_id(model, args.num_mc_samples, ens)
                        output, output_logits = model(input, return_both=True)
                        embedding_b += output/num_ens
                        mis[i] = (mis[i] * ens + (-output_logits.softmax(-1) * output_logits.log_softmax(-1)).sum(1)) / (ens + 1)
                        preds[i] = (preds[i] * ens + output_logits.softmax(-1)) / (ens + 1)

                norm = torch.norm(embedding_b, 2, 1, True)
                embedding = torch.div(embedding_b, norm)
                embeddings.append(embedding)

    embeddings = torch.cat(embeddings).data.cpu().numpy()
    preds = torch.cat(preds, 0)
    mis = (- preds * preds.log()).sum(1) - (0 if num_ens == 1 else torch.cat(mis, 0))
    if (isinstance(val_loaders, list) and args.gpu < len(val_loaders)) or ((not isinstance(val_loaders, list)) and args.gpu == 0):
        np.save(os.path.join(args.save_path, 'mis_{}.npy'.format(name)), mis.data.cpu().numpy())
    if issame is not None:
        tpr, fpr, accuracy, best_thresholds = verify(embeddings, issame, 10)
        print_log('  **Test** {}: {:.3f}'.format(name, accuracy.mean()), log, True)
    torch.distributed.barrier()

def ens_attack(val_loaders, model, criterion, args, log, num_ens=20, amp_autocast=suppress):
    def _grad(X, y, mean, std):
        with model.no_sync():
            with torch.enable_grad():
                X.requires_grad_()
                if args.bayes == 'low_rank':
                    sample_id = np.tile(np.arange(num_ens_a)[:, None], (1, X.size(0))).reshape(-1)
                    set_mc_sample_id(model,num_ens_a, sample_id)
                output = model(X.sub(mean).div(std).repeat(num_ens_a, 1, 1, 1), True)
                output = output.reshape(num_ens_a, X.size(0)//2, 2, output.size(-1))
                loss = ((output[:, :, 0, :].mean(0) - y[:, 1, :].detach())**2).sum(1) + ((output[:, :, 1, :].mean(0) - y[:, 0, :].detach())**2).sum(1)
                grad_ = torch.autograd.grad(
                    [loss], [X], grad_outputs=torch.ones_like(loss), retain_graph=False)[0].detach()
        return grad_

    def _pgd_whitebox(X, mean, std):
        model.apply(freeze)
        y = model(X.sub(mean).div(std), True).reshape(X.size(0)//2, 2, -1)
        model.apply(unfreeze)

        X_pgd = X.clone()
        if args.random: X_pgd += torch.cuda.FloatTensor(*X_pgd.shape).uniform_(-args.epsilon, args.epsilon)

        for _ in range(args.num_steps):
            grad_ = _grad(X_pgd, y, mean, std)
            X_pgd += args.step_size * grad_.sign()
            eta = torch.clamp(X_pgd - X, -args.epsilon, args.epsilon)
            X_pgd = torch.clamp(X + eta, 0, 1.0)

        mis = 0
        preds = 0
        embedding_b = 0
        with amp_autocast():
            for ens in range(num_ens):
                if args.bayes == 'low_rank':
                    set_mc_sample_id(model, args.num_mc_samples, ens)
                output, output_logits = model(X_pgd.sub(mean).div(std), return_both=True)
                embedding_b += output/num_ens
                mis = (mis * ens + (-output_logits.softmax(-1) * (output_logits).log_softmax(-1)).sum(1)) / (ens + 1)
                preds = (preds * ens + output_logits.softmax(-1)) / (ens + 1)

            norm = torch.norm(embedding_b, 2, 1, True)
            embedding = torch.div(embedding_b, norm)
            mis = (- preds * (preds+1e-8).log()).sum(1) - (0 if num_ens == 1 else mis)
        return embedding, mis

    mean = torch.from_numpy(np.array([0.5, 0.5, 0.5])).view(1,3,1,1).cuda(args.gpu).float()
    std = torch.from_numpy(np.array([0.5, 0.5, 0.5])).view(1,3,1,1).cuda(args.gpu).float()
    num_ens_a = min(8, num_ens)

    model.eval()
    if args.dropout_rate > 0.:
        for m in model.modules():
            if m.__class__.__name__.startswith('Dropout'): m.train()
    name, val_loader, issame = val_loaders[args.gpu % len(val_loaders)]
    with torch.no_grad():
        with model.no_sync():
            mis = []
            embeddings = []
            for i, input in enumerate(val_loader):
                is_pair = issame[i*args.batch_size//4:min(len(issame), i*args.batch_size//4+args.batch_size//4)]
                if np.all(is_pair == False): continue

                input = input.cuda(args.gpu, non_blocking=True).mul_(std).add_(mean)
                input = input.reshape(args.batch_size//4, 2, 3, 112, 112)
                assert len(is_pair) == input.shape[0], (len(is_pair), input.shape[0])

                mask = torch.from_numpy(is_pair).cuda(args.gpu, non_blocking=True) == True
                input = input[mask, :, :, :, :].view(-1, 3, 112, 112)

                embedding, mis_ = _pgd_whitebox(input, mean, std)
                mis.append(mis_)
                embeddings.append(embedding)

            mis = torch.cat(mis, 0)
            # embeddings = torch.cat(embeddings, 0).data.cpu().numpy()
            # tpr, fpr, accuracy, best_thresholds = verify(embeddings, np.ones(len(embeddings)).astype(bool), 10)
            # print_log('  **Test** {}: {:.3f}'.format(name, accuracy.mean()), log, True)

    torch.distributed.barrier()
    if args.gpu < len(val_loaders): np.save(os.path.join(args.save_path, 'mis_adv_{}.npy'.format(name)), mis.data.cpu().numpy())

def print_log(print_string, log, force=False):
    if log[1] == 0 or force:
        print("{}".format(print_string))
        log[0].write('{}\n'.format(print_string))
        log[0].flush()

def save_checkpoint(state, is_best, save_path, filename):
    filename = os.path.join(save_path, filename)
    torch.save(state, filename)
    if is_best:
        bestname = os.path.join(save_path, 'model_best.pth.tar')
        shutil.copyfile(filename, bestname)

def adjust_learning_rate_per_step(optimizer, epoch, i, num_ites_per_epoch, args):
    optimizer.param_groups[0]['lr'] = args.ft_lr# * (1 + math.cos(math.pi * (epoch + float(i)/num_ites_per_epoch) / args.epochs)) / 2.
    optimizer.param_groups[1]['lr'] = args.lr_min + (args.lr-args.lr_min) * (1 + math.cos(math.pi * (epoch + float(i)/num_ites_per_epoch) / args.epochs)) / 2.
    return optimizer.param_groups[0]['lr'], optimizer.param_groups[1]['lr']

def adjust_learning_rate(optimizer, epoch, args):
    if epoch in args.schedule:
        scale = args.gammas[args.schedule.index(epoch)]
        for param_group in optimizer.param_groups:
            param_group['lr'] *= scale
            param_group['lr'] = max(param_group['lr'], args.lr_min)
    return tuple([param_group['lr'] for param_group in optimizer.param_groups])

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

if __name__ == '__main__': main()
