from __future__ import division
import time
import numpy as np
import argparse

import torch
import torch.backends.cudnn as cudnn

import models.resnet as models
from mean_field import *

model_names = sorted(name for name in models.__dict__ if name.islower() and not name.startswith("__") and callable(models.__dict__[name]))

def main():
    args = argparse.ArgumentParser(description='tmp', formatter_class=argparse.ArgumentDefaultsHelpFormatter).parse_args()
    args.bayes = 'mf'
    args.log_sigma_init_range = [-6, -5]
    args.single_eps = False
    args.local_reparam = False
    args.dropout_rate = 0
    net = models.__dict__['resnet50'](args)
    net = net.cuda()
    input = torch.cuda.FloatTensor(32, 3, 224, 224).uniform_()
    target = torch.cuda.LongTensor(32).random_(1000)

    if False:
        # test speed
        cudnn.benchmark = True

        ## bayes timing
        train_time, eval_time = [], []
        net.apply(unfreeze)

        net.train()
        torch.cuda.synchronize()
        torch.cuda.synchronize()
        for _ in range(200):
            start_time = time.perf_counter()
            output = net(input)
            loss = torch.nn.functional.cross_entropy(output, target)
            loss.backward()
            torch.cuda.synchronize()
            if _ >= 100:
                train_time.append(time.perf_counter() - start_time)
        print('BayesAdapter train on GPU: {:.02e}s/ite'.format(np.mean(train_time)))

        net.eval()
        torch.cuda.synchronize()
        torch.cuda.synchronize()
        with torch.no_grad():
            for _ in range(200):
                start_time = time.perf_counter()
                output = net(input)
                torch.cuda.synchronize()
                if _ >= 100:
                    eval_time.append(time.perf_counter() - start_time)
        print('BayesAdapter eval on GPU: {:.02e}s/ite'.format(np.mean(eval_time)))

        ## local reparameterization timing
        train_time, eval_time = [], []
        net.apply(unfreeze)
        net.apply(enable_lrt)

        net.train()
        torch.cuda.synchronize()
        torch.cuda.synchronize()
        for _ in range(200):
            start_time = time.perf_counter()
            output = net(input)
            loss = torch.nn.functional.cross_entropy(output, target)
            loss.backward()
            torch.cuda.synchronize()
            if _ >= 100:
                train_time.append(time.perf_counter() - start_time)
        print('Local reparameterization train on GPU: {:.02e}s/ite'.format(np.mean(train_time)))

        net.eval()
        torch.cuda.synchronize()
        torch.cuda.synchronize()
        with torch.no_grad():
            for _ in range(200):
                start_time = time.perf_counter()
                output = net(input)
                torch.cuda.synchronize()
                if _ >= 100:
                    eval_time.append(time.perf_counter() - start_time)
        print('Local reparameterization eval on GPU: {:.02e}s/ite'.format(np.mean(eval_time)))
        net.apply(disable_lrt)

        ## deterministic timing
        train_time, eval_time = [], []
        net.apply(freeze)

        net.train()
        torch.cuda.synchronize()
        torch.cuda.synchronize()
        for _ in range(200):
            start_time = time.perf_counter()
            output = net(input)
            loss = torch.nn.functional.cross_entropy(output, target)
            loss.backward()
            torch.cuda.synchronize()
            if _ >= 100:
                train_time.append(time.perf_counter() - start_time)
        print('Deterministic train on GPU: {:.02e}s/ite'.format(np.mean(train_time)))

        net.eval()
        torch.cuda.synchronize()
        torch.cuda.synchronize()
        with torch.no_grad():
            for _ in range(200):
                start_time = time.perf_counter()
                output = net(input)
                torch.cuda.synchronize()
                if _ >= 100:
                    eval_time.append(time.perf_counter() - start_time)
        print('Deterministic eval on GPU: {:.02e}s/ite'.format(np.mean(eval_time)))
    else:
        # test grad var
        conv = BayesConv2dMF(single_eps=False, local_reparam=False, in_channels=3, out_channels=16, kernel_size=3, padding=1, bias=False).cuda()
        torch.nn.init.kaiming_normal_(conv.weight_mu, mode='fan_out', nonlinearity='relu')
        conv.weight_log_sigma.data.uniform_(-6, -5)
        input = torch.cuda.FloatTensor(128, 3, 32, 32).uniform_()
        target = torch.cuda.FloatTensor(128, 16, 32, 32).uniform_()

        k1_grads = []
        for _ in range(500):
            loss = ((conv(input) - target)**2).mean()
            loss.backward()
            k1_grads.append(torch.cat([conv.weight_mu.grad.data.view(-1), conv.weight_log_sigma.grad.data.view(-1)]).cpu().numpy())
            conv.weight_mu.grad.zero_()
            conv.weight_log_sigma.grad.zero_()

        conv.local_reparam = True
        k1_5_grads = []
        for _ in range(500):
            loss = ((conv(input) - target)**2).mean()
            loss.backward()
            k1_5_grads.append(torch.cat([conv.weight_mu.grad.data.view(-1), conv.weight_log_sigma.grad.data.view(-1)]).cpu().numpy())
            conv.weight_mu.grad.zero_()
            conv.weight_log_sigma.grad.zero_()

        conv.single_eps = True
        k2_grads = []
        for _ in range(500):
            loss = ((conv(input) - target)**2).mean()
            loss.backward()
            k2_grads.append(torch.cat([conv.weight_mu.grad.data.view(-1), conv.weight_log_sigma.grad.data.view(-1)]).cpu().numpy())
            conv.weight_mu.grad.zero_()
            conv.weight_log_sigma.grad.zero_()

        k1_grads = np.stack(k1_grads)
        k1_5_grads = np.stack(k1_5_grads)
        k2_grads = np.stack(k2_grads)
        print(k1_grads.shape, k1_5_grads.shape, k2_grads.shape)
        k1_grads_vars = np.var(k1_grads, axis=0, ddof=1)
        k1_5_grads_vars = np.var(k1_5_grads, axis=0, ddof=1)
        k2_grads_vars = np.var(k2_grads, axis=0, ddof=1)
        print(k1_grads_vars[:len(k1_grads_vars)//2].mean(), k1_grads_vars[len(k1_grads_vars)//2:].mean())
        print(k1_5_grads_vars[:len(k1_5_grads_vars)//2].mean(), k1_5_grads_vars[len(k1_5_grads_vars)//2:].mean())
        print(k2_grads_vars[:len(k2_grads_vars)//2].mean(), k2_grads_vars[len(k2_grads_vars)//2:].mean())

        import pandas as pd
        from sklearn.decomposition import PCA
        from sklearn.manifold import TSNE
        import matplotlib
        matplotlib.use('agg')
        import matplotlib.pyplot as plt
        plt.rcParams.update({'figure.max_open_warning': 0})
        plt.rcParams['font.family'] = 'Times New Roman'
        plt.rcParams['font.size'] = '14'
        import seaborn as sns

        X = np.concatenate([k1_grads[:, :k1_grads.shape[1]//2], k1_5_grads[:, :k1_5_grads.shape[1]//2], k2_grads[:, :k2_grads.shape[1]//2]])
        y = np.zeros(X.shape[0])
        y[y.shape[0]//3:y.shape[0]//3*2] = 1
        y[y.shape[0]//3*2:] = 2
        feat_cols = [ 'grad_dim_'+str(i) for i in range(X.shape[1]) ]
        df = pd.DataFrame(X, columns=feat_cols)
        df['y'] = y

        def assign_label(i):
            if i == 0:
                return 'exemplar reparameterization'
            elif i == 1:
                return 'local reparameterization'
            else:
                return 'vanilla'
        df['label'] = df['y'].apply(assign_label)

        # pca = PCA(n_components=3)
        # pca_result = pca.fit_transform(df[feat_cols].values)
        # df['pca-one'] = pca_result[:,0]
        # df['pca-two'] = pca_result[:,1]
        # df['pca-three'] = pca_result[:,2]
        # plt.figure(figsize=(8,8))
        # sns.scatterplot(
        #     x="pca-one", y="pca-two",
        #     hue="label",
        #     palette=sns.color_palette("hls", 2),
        #     data=df,
        #     legend="full",
        #     alpha=1
        # )
        # plt.xlabel('')
        # plt.ylabel('')
        # handles, labels = plt.gca().get_legend_handles_labels()
        # plt.legend(handles=handles[1:], labels=labels[1:])
        # plt.tight_layout()
        # plt.savefig('grads_var_pca2.pdf', format='pdf', bbox_inches='tight')

        tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
        tsne_results = tsne.fit_transform(df[feat_cols].values)
        df['tsne-2d-one'] = tsne_results[:,0]
        df['tsne-2d-two'] = tsne_results[:,1]
        plt.figure(figsize=(8,8))
        sns.scatterplot(
            x="tsne-2d-one", y="tsne-2d-two",
            hue="label",
            palette=sns.color_palette("hls", 3),
            data=df,
            legend="full",
            alpha=1
        )
        plt.xlabel('')
        plt.ylabel('')
        handles, labels = plt.gca().get_legend_handles_labels()
        plt.legend(handles=handles[1:], labels=labels[1:])
        plt.tight_layout()
        plt.savefig('grads_var_tsne2.pdf', format='pdf', bbox_inches='tight')

        # pca_50 = PCA(n_components=50)
        # pca_result_50 = pca_50.fit_transform(df[feat_cols].values)
        # tsne = TSNE(n_components=2, verbose=0, perplexity=40, n_iter=300)
        # tsne_pca_results = tsne.fit_transform(pca_result_50)
        # df['tsne-pca50-one'] = tsne_pca_results[:,0]
        # df['tsne-pca50-two'] = tsne_pca_results[:,1]
        # plt.figure(figsize=(8,8))
        # sns.scatterplot(
        #     x="tsne-pca50-one", y="tsne-pca50-two",
        #     hue="label",
        #     palette=sns.color_palette("hls", 2),
        #     data=df,
        #     legend="full",
        #     alpha=1
        # )
        # plt.xlabel('')
        # plt.ylabel('')
        # handles, labels = plt.gca().get_legend_handles_labels()
        # plt.legend(handles=handles[1:], labels=labels[1:])
        # plt.tight_layout()
        # plt.savefig('grads_var_pca-tsne.pdf', format='pdf', bbox_inches='tight')


        # X = np.concatenate((k1_grads[:, k1_grads.shape[1]//2:], k2_grads[:, k2_grads.shape[1]//2:]))
        # y = np.zeros(X.shape[0])
        # y[y.shape[0]//2:] = 1
        # feat_cols = [ 'grad_dim_'+str(i) for i in range(X.shape[1]) ]
        # df = pd.DataFrame(X, columns=feat_cols)
        # df['y'] = y
        # df['label'] = df['y'].apply(lambda i: 'variance reduced' if i == 0 else 'vanilla')
        #
        # tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
        # tsne_results = tsne.fit_transform(df[feat_cols].values)
        # df['tsne-2d-one'] = tsne_results[:,0]
        # df['tsne-2d-two'] = tsne_results[:,1]
        # plt.figure(figsize=(8,8))
        # sns.scatterplot(
        #     x="tsne-2d-one", y="tsne-2d-two",
        #     hue="label",
        #     palette=sns.color_palette("hls", 2),
        #     data=df,
        #     legend="full",
        #     alpha=1
        # )
        # plt.xlabel('')
        # plt.ylabel('')
        # handles, labels = plt.gca().get_legend_handles_labels()
        # plt.legend(handles=handles[1:], labels=labels[1:])
        # plt.tight_layout()
        # plt.savefig('grads_var_tsne2_logsigma.pdf', format='pdf', bbox_inches='tight')

def freeze(m):
    if isinstance(m, (BayesConv2dMF, BayesLinearMF, BayesBatchNorm2dMF)):
        m.deterministic = True

def unfreeze(m):
    if isinstance(m, (BayesConv2dMF, BayesLinearMF, BayesBatchNorm2dMF)):
        m.deterministic = False

def enable_lrt(m):
    if isinstance(m, (BayesConv2dMF, BayesLinearMF, BayesBatchNorm2dMF)):
        m.local_reparam = True

def disable_lrt(m):
    if isinstance(m, (BayesConv2dMF, BayesLinearMF, BayesBatchNorm2dMF)):
        m.local_reparam = False

if __name__ == '__main__': main()
