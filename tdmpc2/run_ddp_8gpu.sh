#!/bin/bash

#########################################################################
# Distributed Data Parallel (DDP) Training Script for TD-MPC2
#
# This script launches 8-GPU training for TD-MPC2 with online RL
#
# Usage:
#   ./run_ddp_8gpu.sh [task_name] [model_size] [additional_args]
#
# Examples:
#   ./run_ddp_8gpu.sh dog-run 5
#   ./run_ddp_8gpu.sh humanoid-walk 19 steps=20000000
#   ./run_ddp_8gpu.sh walker-run 5 sync_freq=2
#########################################################################

# Default values
TASK=${1:-"dog-run"}
MODEL_SIZE=${2:-5}
WORLD_SIZE=${3:-8}
SYNC_FREQ=${4:-1}

# Shift to get additional arguments
shift 4 2>/dev/null || shift $#

echo "=========================================="
echo "TD-MPC2 DDP Training Configuration"
echo "=========================================="
echo "Task:           $TASK"
echo "Model Size:     $MODEL_SIZE"
echo "World Size:     $WORLD_SIZE GPUs"
echo "Sync Frequency: $SYNC_FREQ"
echo "Additional Args: $@"
echo "=========================================="
echo ""

# Check if CUDA is available
if ! command -v nvidia-smi &> /dev/null; then
    echo "Error: nvidia-smi not found. CUDA may not be available."
    exit 1
fi

# Check number of available GPUs
NUM_GPUS=$(nvidia-smi --list-gpus | wc -l)
echo "Detected $NUM_GPUS GPUs"

if [ "$WORLD_SIZE" -gt "$NUM_GPUS" ]; then
    echo "Error: Requested $WORLD_SIZE GPUs but only $NUM_GPUS available"
    exit 1
fi

# Set CUDA devices (use first WORLD_SIZE GPUs)
export CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((WORLD_SIZE-1)))
echo "Using GPUs: $CUDA_VISIBLE_DEVICES"
echo ""

# Run training
python train_ddp.py \
    task=$TASK \
    model_size=$MODEL_SIZE \
    world_size=$WORLD_SIZE \
    sync_freq=$SYNC_FREQ \
    "$@"

echo ""
echo "=========================================="
echo "Training completed!"
echo "=========================================="
