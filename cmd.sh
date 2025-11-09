#!/usr/bin/env bash

set -euo pipefail

export NCCL_P2P_DISABLE=0
export NCCL_SHM_DISABLE=0
export NCCL_SOCKET_IFNAME=lo
export NCCL_IB_DISABLE=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

export OMP_NUM_THREADS=$(nproc)
export MKL_NUM_THREADS=$(nproc)
export OPENBLAS_NUM_THREADS=$(nproc)


CONFIG="${1:-qwen3_pp}"
case "$CONFIG" in
  qwen3_pp|gpt1b_pp|gpt1b_pp_resume|gpt1b_pp_expert_ft)
    shift
    ;;
  *)
    CONFIG="qwen3_pp"
    ;;
esac

EXTRA_ARGS=("$@")

case "$CONFIG" in
  qwen3_pp)
    torchrun --nproc_per_node=8 \
      scripts/train_pipeline_ddp.py \
      --model qwen3 \
      --env humanoid \
      --dataset medium \
      --pipeline_stages 4 \
      --data_parallel_groups 2 \
      --device_groups '0,1,2,3;4,5,6,7' \
      --batch_size 128 \
      --micro_batches 16 \
      --n_blocks 20 \
      --embed_dim 1536 \
      --n_heads 12 \
      --num_kv_heads 6 \
      --head_dim 128 \
      --lr 8e-5 \
      --warmup_steps 12000 \
      --grad_clip 0.25 \
      --max_train_iters 100 \
      --num_updates_per_iter 1000 \
      --eval_interval 1 \
      --progress_refresh 2.0 \
      --log_dir dt_runs/qwen3_pp \
      --dist_backend nccl "${EXTRA_ARGS[@]}"
    ;;
  gpt1b_pp)
    torchrun --nproc_per_node=8 \
      scripts/train_pipeline_ddp.py \
      --model dt \
      --env humanoid \
      --dataset medium \
      --pipeline_stages 4 \
      --data_parallel_groups 2 \
      --device_groups '0,1,2,3;4,5,6,7' \
      --batch_size 64 \
      --micro_batches 16 \
      --n_blocks 20 \
      --embed_dim 1536 \
      --n_heads 24 \
      --dropout_p 0.1 \
      --lr 8e-5 \
      --warmup_steps 12000 \
      --grad_clip 0.25 \
      --max_train_iters 100 \
      --num_updates_per_iter 1000 \
      --eval_interval 5 \
      --progress_refresh 2.0 \
      --log_dir dt_runs/gpt1b_pp \
      --dist_backend nccl "${EXTRA_ARGS[@]}"
    ;;
  gpt1b_pp_resume)
    torchrun --nproc_per_node=8 \
      scripts/train_pipeline_ddp.py \
      --model dt \
      --env humanoid \
      --dataset medium \
      --pipeline_stages 4 \
      --data_parallel_groups 2 \
      --device_groups '0,1,2,3;4,5,6,7' \
      --batch_size 64 \
      --micro_batches 16 \
      --n_blocks 20 \
      --embed_dim 1536 \
      --n_heads 24 \
      --dropout_p 0.1 \
      --lr 8e-5 \
      --warmup_steps 12000 \
      --grad_clip 0.25 \
      --max_train_iters 100 \
      --num_updates_per_iter 1000 \
      --eval_interval 5 \
      --progress_refresh 2.0 \
      --log_dir dt_runs/gpt1b_pp \
      --resume_checkpoint dt_runs/gpt1b_pp/dtpp_humanoid-medium-v5_ckpt_iter00003_updates00003000.pt \
      --dist_backend nccl "${EXTRA_ARGS[@]}"
    ;;
  gpt1b_pp_expert_ft)
    torchrun --nproc_per_node=8 \
      scripts/train_pipeline_ddp.py \
      --model dt \
      --env humanoid \
      --dataset expert \
      --pipeline_stages 4 \
      --data_parallel_groups 2 \
      --device_groups '0,1,2,3;4,5,6,7' \
      --batch_size 64 \
      --micro_batches 16 \
      --n_blocks 20 \
      --embed_dim 1536 \
      --n_heads 24 \
      --dropout_p 0.1 \
      --lr 1e-6 \
      --warmup_steps 2000 \
      --grad_clip 0.25 \
      --max_train_iters 100 \
      --num_updates_per_iter 1000 \
      --eval_interval 1 \
      --progress_refresh 2.0 \
      --log_dir dt_runs/gpt1b_pp_expert_ft2 \
      --resume_checkpoint dt_runs/gpt1b_pp_expert_ft/dtpp_humanoid-expert-v5_ckpt_iter00036_updates00036000.pt \
      --dist_backend nccl "${EXTRA_ARGS[@]}"
    ;;
esac
