#!/usr/bin/env bash

# Training script
PATH_TO_DATASET="/run/media/stec/Box4T/open_images"
# PATH_TO_VALIDATION_DATASET="/run/media/stec/Box4T/SSK/Dataset/samples/test_images"
PATH_TO_STORE_CHECKPOINTS="models"

python train_ssd.py \
--dataset_type open_images \
--datasets $PATH_TO_DATASET \ 
# --validation_dataset $PATH_TO_DATASET \ 
--net mb2-ssd-lite \
--pretrained_ssd models/mb2-ssd-lite-mp-0_686.pth \  
--scheduler cosine \
--use_cuda \
--batch_size 20 \
--num_workers 10 \
--lr 0.05 \
--t_max 100 \
--validation_epochs 5 \ 
--num_epochs 100 \
--base_net_lr 0.001 \
--checkpoint_folder $PATH_TO_STORE_CHECKPOINTS

# python train_ssd.py 
# --dataset_type voc  
# --datasets ~/data/VOC0712/VOC2007 ~/data/VOC0712/VOC2012 
# --validation_dataset ~/data/VOC0712/test/VOC2007/ 
# --net mb2-ssd-lite 
# --base_net models/mb2-imagenet-71_8.pth  
# --scheduler cosine 
# --lr 0.01 
# --t_max 200 
# --validation_epochs 5 
# --num_epochs 200
