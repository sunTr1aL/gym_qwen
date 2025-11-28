#!/usr/bin/env bash
set -euo pipefail

python train.py \
  task=mujoco-walker obs=state model_size=5 offline=false episodic=true \
  enable_wandb=true save_video=true save_agent=true \
  wandb_project=tdmpc2-walker wandb_entity=lzy66632-university-of-illinois-urbana-champaign \
  compile=false
