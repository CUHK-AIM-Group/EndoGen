# # !/bin/bash
# set -x

# #!/bin/bash

CUDA_VISIBLE_DEVICES=7 torchrun \
--nnodes=1 --nproc_per_node=1 --node_rank=0 \
--master_port=12347 \
sample_c2i.py \
--vq-ckpt ./pretrained_models/vq_ds16_c2i.pt \
--gpt-ckpt /data_hdd/xyliu/medical_videos/surgvisdom/train/Porcine/frames_4x4_128_code_trained_models/000-GPT-B/checkpoints/0065324.pt \
--gpt-model GPT-B --image-size 512 --cfg-scale 1.0 \
--save-folder /data_hdd/xyliu/medical_videos/surgvisdom/train/Porcine/frames_4x4_128_code_trained_models/000-GPT-B/samples/0065324 \
--seed $((i + 2)) \
--name "run_$i" \
--num-classes 3 \
"$@"