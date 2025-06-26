# # !/bin/bash
# set -x

# #!/bin/bash

CUDA_VISIBLE_DEVICES=2 torchrun \
--nnodes=1 --nproc_per_node=1 --node_rank=0 \
--master_port=12345 \
sample_c2i_demo.py \
--vq-ckpt ./pretrained_models/vq_ds16_c2i.pt \
--gpt-ckpt ./pretrained_models/hyperkvasir_model_0054236.pt \
--gpt-model GPT-B --image-size 512 --cfg-scale 1.0 \
--save-folder ./demo_endogen \
--seed 40 \
--name "run_$i" \
--num-classes 8 \
"$@"