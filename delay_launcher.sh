#!/bin/bash
printf "%s\n" "waiting..."
/bin/sleep 14400
printf "%s\n" "prepare env"
cd /data/zhijie/autobayes
source /home/zhijie/env3/bin/activate
printf "%s\n" "start"
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python finetune.py --cutout  --job-id test --bayes mf --log_sigma_init_range -6 -5 --epochs 100 --max_choice 1 --schedule 5 10 15 --batch_size 256
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python main.py --cutout --job-id map-decay2-dp0.3 --batch_size 256 --decay 0.0002 --dropout_rate 0.3


# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python finetune.py --cutout  --job-id map-decay2-dp0.3 --evaluate --dropout_rate 0.3 --resume /data/zhijie/snapshots_ab/map-decay2-dp0.3/checkpoint.pth.tar

# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python finetune_in.py --job-id map-pretrained --evaluate

# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python finetune_in.py --job-id map --evaluate --resume /data/zhijie/snapshots_ab_in/map/checkpoint.pth.tar

# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python finetune_in.py --job-id bayes-init6_5-warmup --evaluate --resume /data/zhijie/snapshots_ab_in/bayes-init6_5-warmup/checkpoint.pth.tar --bayes mf

# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python finetune_in.py --job-id map-dp0.2-afterconv --evaluate --resume /data/zhijie/snapshots_ab_in/map-dp0.2-afterconv/checkpoint.pth.tar --dropout_rate 0.2
kill 28167
kill 28168
kill 28169
kill 28170
kill 28171
kill 28172
kill 28173
kill 28174

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python finetune.py --cutout  --job-id ft-gan1000-.55 --bayes mf  --epochs 20 --schedule 5 10 15 --num_gan 1000 --mi_th 0.55 --dist-port 2345
