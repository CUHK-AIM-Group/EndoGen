# !/bin/bash
set -x

CUDA_VISIBLE_DEVICES=2,8 torchrun \
--nnodes=1 --nproc_per_node=2 --master_port=12113 \
train_c2i.py \
--image-size 1024 \
--global-batch-size 8 \
--code-path /data_hdd/xyliu/medical_videos/hyperkvasir/frames_8x8_128_code \
--results-dir /data_hdd/xyliu/medical_videos/hyperkvasir/frames_8x8_128_code_trained_models \
--gpt-ckpt ./pretrained_models/c2i_B_256.pt \
--num-classes 8 \
--token-dropout-p 0.3 \
--tok-drop-method 'adaptive' \
"$@"
# 48