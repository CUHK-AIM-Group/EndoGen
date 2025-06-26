# # !/bin/bash
# set -x

# #!/bin/bash

CUDA_VISIBLE_DEVICES=6 torchrun \
--nnodes=1 --nproc_per_node=1 --node_rank=0 \
--master_port=12365 \
sample_c2i_64frame.py \
--vq-ckpt ./pretrained_models/vq_ds16_c2i.pt \
--gpt-ckpt /data_hdd/xyliu/medical_videos/hyperkvasir/frames_8x8_128_code_trained_models/000-GPT-B/checkpoints/0054236.pt \
--gpt-model GPT-B --image-size 1024 --cfg-scale 1.0 \
--save-folder /data_hdd/xyliu/medical_videos/hyperkvasir/frames_8x8_128_code_trained_models/000-GPT-B/samples/0054236 \
--seed $((i + 2)) \
--name "run_$i" \
--num-classes 8 \
"$@"