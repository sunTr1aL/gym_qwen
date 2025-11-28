#!/bin/bash

#########################################################################
# Example DDP Training Commands for TD-MPC2
#
# This script contains various example configurations for different
# use cases. Uncomment the one you want to run.
#########################################################################

# Example 1: Basic 8-GPU training on dog-run task
# python train_ddp.py task=dog-run model_size=5 world_size=8

# Example 2: Humanoid tasks with large model
# python train_ddp.py task=humanoid-walk model_size=19 world_size=8 steps=15000000

# Example 3: Fast training with gradient accumulation
# python train_ddp.py task=walker-run world_size=8 sync_freq=2 batch_size=256

# Example 4: Memory-constrained setup (small batch size)
# python train_ddp.py task=dog-run world_size=8 batch_size=128 buffer_size=500000

# Example 5: High-throughput training (large batch size)
# python train_ddp.py task=cheetah-run world_size=8 batch_size=512

# Example 6: 4-GPU training
# python train_ddp.py task=walker-walk world_size=4 model_size=5

# Example 7: Training with custom evaluation frequency
# python train_ddp.py task=dog-run world_size=8 eval_freq=25000

# Example 8: Long training run with checkpointing
# python train_ddp.py task=humanoid-run world_size=8 steps=30000000 save_agent=true

# Example 9: Training without video saving (faster)
# python train_ddp.py task=walker-run world_size=8 save_video=false

# Example 10: Custom wandb project
# python train_ddp.py task=dog-run world_size=8 enable_wandb=true wandb_project=my-experiments

# Example 11: All humanoid tasks (run separately)
echo "Training humanoid tasks with 8 GPUs..."

# Uncomment the tasks you want to train:
# python train_ddp.py task=humanoid-walk model_size=19 world_size=8 exp_name=humanoid-walk-ddp
# python train_ddp.py task=humanoid-run model_size=19 world_size=8 exp_name=humanoid-run-ddp
# python train_ddp.py task=humanoid-stand model_size=19 world_size=8 exp_name=humanoid-stand-ddp

# Example 12: Testing different sync frequencies
echo "Testing sync frequency impact..."

# Run with sync_freq=1 (baseline)
# python train_ddp.py task=walker-run world_size=8 sync_freq=1 exp_name=sync-1 steps=1000000

# Run with sync_freq=2
# python train_ddp.py task=walker-run world_size=8 sync_freq=2 exp_name=sync-2 steps=1000000

# Run with sync_freq=4
# python train_ddp.py task=walker-run world_size=8 sync_freq=4 exp_name=sync-4 steps=1000000

# Example 13: Sweep over model sizes
echo "Model size comparison..."

# Small model (5M params)
# python train_ddp.py task=dog-run model_size=5 world_size=8 exp_name=size-5

# Medium model (19M params)
# python train_ddp.py task=dog-run model_size=19 world_size=8 exp_name=size-19

# Large model (48M params)
# python train_ddp.py task=dog-run model_size=48 world_size=8 exp_name=size-48

# Example 14: Episodic task training
# python train_ddp.py task=walker-walk world_size=8 episodic=true

# Example 15: Custom master port (if default is in use)
# python train_ddp.py task=dog-run world_size=8 master_port=12356

echo "Examples script loaded. Uncomment desired commands and run."
