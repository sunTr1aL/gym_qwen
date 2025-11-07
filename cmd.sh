#!/usr/bin/env bash

set -euo pipefail

export NCCL_P2P_DISABLE=0
export NCCL_SHM_DISABLE=0
export NCCL_SOCKET_IFNAME=lo
export NCCL_IB_DISABLE=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

torchrun --nproc_per_node=8 \
  scripts/train_pipeline_ddp.py \
  --model qwen3 \
  --env humanoid \
  --dataset medium \
  --pipeline_stages 4 \
  --data_parallel_groups 2 \
  --device_groups '0,1,2,3;4,5,6,7' \
  --batch_size 32 \
  --micro_batches 4 \
  --n_blocks 24 \
  --embed_dim 1024 \
  --n_heads 16 \
  --num_kv_heads 8 \
  --head_dim 64 \
  --lr 1e-4 \
  --warmup_steps 10000 \
  --grad_clip 0.25 \
  --eval_interval 1 \
  --progress_refresh 2.0 \
  --log_dir dt_runs/qwen3_pp \
  --dist_backend nccl "$@"
