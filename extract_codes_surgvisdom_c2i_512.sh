# !/bin/bash
set -x

CUDA_VISIBLE_DEVICES=7 torchrun \
--nnodes=1 --nproc_per_node=1 --node_rank=0 \
--master_port=12335 \
extract_codes_c2i.py \
--vq-model VQ-16 \
--vq-ckpt ./pretrained_models/vq_ds16_c2i.pt \
--data-path /data_hdd/xyliu/medical_videos/surgvisdom/train/Porcine/frames_4x4_128 \
--code-path /data_hdd/xyliu/medical_videos/surgvisdom/train/Porcine/frames_4x4_128_code \
--image-size 512 \
"$@"