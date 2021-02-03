# BayesAdapter

For Cifar-10:

Deterministic pre-training:
  >> CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python main.py --cutout --job-id map-decay2 --batch_size 256 --decay 0.0002
  
Bayesian fine-tuning (MFG):
  >> CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python finetune.py --data_path /data/LargeData/Regular/cifar/ --cutout  --job-id ft-mf-mc20 --alpha 0 --dist-port 2345 --amp --finetune --bayes mf --num_mc_samples 20 --epochs 12 --lr 1e-3 --ft_lr 1e-4 --cos_lr

Bayesian fine-tuning (PSE):
  >> CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python finetune.py --data_path /data/LargeData/Regular/cifar/ --cutout  --job-id ft-lr-r1-mc20 --alpha 0 --dist-port 2345 --amp --finetune --bayes low_rank --low_rank 1 --num_mc_samples 20 --epochs 12 --lr 1e-3 --ft_lr 1e-4 --cos_lr

For ImageNet:

Download the deterministic checkpoint of ResNet-50 trained on ImageNet from https://download.pytorch.org/models/resnet50-19c8e357.pth

Bayesian fine-tuning (MFG):
  >> CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python finetune.py --dataset imagenet --arch resnet50 --decay 1e-4 --data_path /data/LargeData/Large/ImageNet/  --job-id in-ft-mf --alpha 0 --dist-port 2345 --amp --finetune --bayes mf --epochs 4 --cos_lr --lr 1e-3 --ft_lr 1e-4

Bayesian fine-tuning (PSE):
  >> CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python finetune.py --dataset imagenet --arch resnet50 --decay 1e-4 --data_path /data/LargeData/Large/ImageNet/  --job-id in-ft-lr-r1-mc20 --alpha 0 --dist-port 2345 --amp --finetune --bayes low_rank --low_rank 1 --epochs 4 --cos_lr --lr 1e-3 --ft_lr 1e-4 --pert_init_std 0.1


For face:

Deterministic pre-training:
  >> CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python main_face.py  --job-id map-ft0.01-drop0-decay5 --dist-port 2345 --ft_learning_rate 0.01 --epochs 90 --schedule 30 60 80 --dropout_rate 0 --decay 5e-4
  
Bayesian fine-tuning (MFG):
  >> CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python finetune_face.py --amp --decay 5e-4 --bayes mf --epochs 4 --alpha 0 --job-id face-ft-mf-mc20 --lr 1e-3 --ft_lr 1e-4 --cos_lr

Bayesian fine-tuning (PSE):
  >> CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python finetune_face.py --amp --decay 5e-4 --bayes low_rank --low_rank 1 --epochs 4 --alpha 0 --job-id face-ft-lr-r1-mc20 --lr 1e-3 --ft_lr 1e-4 --cos_lr --pert_init_std 0.1
