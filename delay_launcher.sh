#!/bin/bash
printf "%s\n" "prepare env"
cd /data/zhijie/autobayes
source /home/zhijie/env3/bin/activate
printf "%s\n" "waiting..."
/bin/sleep 10800
printf "%s\n" "start"
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python finetune.py --cutout  --job-id test --bayes mf --log_sigma_init_range -6 -5 --epochs 100 --max_choice 1 --schedule 5 10 15 --batch_size 256
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python main.py --cutout --job-id map-decay2-dp0.3 --batch_size 256 --decay 0.0002 --dropout_rate 0.3


# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python finetune.py --cutout  --job-id map-decay2-dp0.3 --evaluate --dropout_rate 0.3 --resume /data/zhijie/snapshots_ab/map-decay2-dp0.3/checkpoint.pth.tar

# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python finetune_in.py --job-id map-pretrained --evaluate

# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python finetune_in.py --job-id map --evaluate --resume /data/zhijie/snapshots_ab_in/map/checkpoint.pth.tar

# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python finetune_in.py --job-id bayes-init6_5-warmup --evaluate --resume /data/zhijie/snapshots_ab_in/bayes-init6_5-warmup/checkpoint.pth.tar --bayes mf

# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python finetune_in.py --job-id map-dp0.2-afterconv --evaluate --resume /data/zhijie/snapshots_ab_in/map-dp0.2-afterconv/checkpoint.pth.tar --dropout_rate 0.2
kill 40471
kill 40472
kill 40473
kill 40474
kill 40475
kill 40476
kill 40477
kill 40478

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python finetune.py --data_path /data/LargeData/Regular/cifar/ --cutout  --job-id train-mf-2 --epochs 200 --schedule  60 120 160 --alpha 0 --dist-port 2345 --ft_lr 0.2 --lr 0.2 --amp --bayes mf --num_mc_samples 20 --manualSeed 2

#CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python finetune.py --cutout  --job-id ft-gan1000-.55 --bayes mf  --epochs 20 --schedule 5 10 15 --num_gan 1000 --mi_th 0.55 --dist-port 2345
