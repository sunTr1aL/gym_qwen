#!/usr/bin/env bash
set -euo pipefail

python train.py \
  task=mujoco-walker obs=state model_size=5 offline=true \
  data_dir=/home/weiland/tdmpc2/datasets/converted \
  transformer_dynamic=true transformer_embed_dim=256 \
  transformer_layers=3 transformer_heads=4 transformer_mlp_ratio=2 \
  transformer_max_seq_len=12 transformer_attn_dropout=0 dropout=0.01 \
  horizon=8 mpc=true iterations=6 num_samples=512 num_elites=64 num_pi_trajs=24 \
  min_std=0.05 max_std=2 temperature=0.5 plan_chunk=1 \
  steps=10000000 batch_size=256 lr=3e-4 buffer_size=1000000 \
  reward_coef=0.1 value_coef=0.1 consistency_coef=20 rho=0.7 tau=0.01 grad_clip_norm=20 \
  num_bins=101 vmin=-10 vmax=10 log_std_min=-10 log_std_max=2 entropy_coef=1e-4 \
  eval_freq=50000 eval_episodes=10 seed=1 \
  enable_wandb=true save_video=true save_agent=true \
  wandb_project=tdmpc2-walker wandb_entity=lzy66632-university-of-illinois-urbana-champaign \
  compile=false
