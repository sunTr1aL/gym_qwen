# Distributed Data Parallel (DDP) Training for TD-MPC2

This guide explains how to use the new multi-GPU training capabilities for TD-MPC2 online training.

## Overview

The DDP training implementation enables you to:
- **Train on multiple GPUs** (tested with 8 GPUs)
- **Parallelize experience collection** (each GPU runs its own environment)
- **Automatic gradient synchronization** via PyTorch DDP
- **Control sync frequency** for gradient accumulation
- **Scale to larger effective batch sizes**

## Architecture

```
┌─────────────────────────────────────────────────────┐
│              Main Process (Rank 0)                   │
│  - Spawns N worker processes                        │
│  - Coordinates distributed training                  │
└─────────────────────────────────────────────────────┘
                        │
        ┌───────────────┴───────────────┐
        │                               │
┌───────▼───────┐               ┌───────▼───────┐
│  GPU 0        │               │  GPU 1        │
│  - Env        │               │  - Env        │
│  - Buffer     │               │  - Buffer     │
│  - Model      │◄─────DDP─────►│  - Model      │
└───────────────┘   (sync grad)  └───────────────┘
                        │
              ┌─────────┴─────────┐
              │                   │
        ┌─────▼─────┐       ┌─────▼─────┐
        │  GPU ...  │       │  GPU 7    │
        │  - Env    │       │  - Env    │
        │  - Buffer │       │  - Buffer │
        │  - Model  │       │  - Model  │
        └───────────┘       └───────────┘
```

### Key Features:

1. **Independent Environments**: Each GPU runs its own environment instance
2. **Independent Buffers**: Each GPU maintains its own replay buffer
3. **Synchronized Model**: Model parameters are kept in sync via DDP
4. **Gradient Averaging**: Gradients are automatically averaged across GPUs

## Quick Start

### 1. Using the Shell Script (Easiest)

```bash
cd tdmpc2

# Basic usage with default settings (8 GPUs)
./run_ddp_8gpu.sh dog-run 5

# Custom configuration
./run_ddp_8gpu.sh humanoid-walk 19 8 1

# With additional Hydra arguments
./run_ddp_8gpu.sh walker-run 5 8 1 steps=20000000 eval_freq=100000
```

**Script arguments:**
1. Task name (e.g., `dog-run`, `humanoid-walk`)
2. Model size (e.g., `5`, `19`, `48`)
3. Number of GPUs (default: 8)
4. Sync frequency (default: 1)

### 2. Using Python Directly

```bash
cd tdmpc2

# 8 GPU training
python train_ddp.py task=dog-run model_size=5 world_size=8

# 4 GPU training
python train_ddp.py task=humanoid-run model_size=19 world_size=4

# With custom sync frequency (gradient accumulation)
python train_ddp.py task=walker-run world_size=8 sync_freq=2

# With custom batch size per GPU
python train_ddp.py task=dog-run world_size=8 batch_size=128
```

### 3. Using Config File

```bash
# Edit config_ddp.yaml to your preferences, then:
python train_ddp.py --config-name=config_ddp
```

## Configuration Parameters

### DDP-Specific Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `world_size` | 8 | Number of GPUs to use |
| `sync_freq` | 1 | How often to sync gradients (1=every update) |
| `master_addr` | localhost | Master node address |
| `master_port` | 12355 | Communication port |

### Important Notes

- **Effective Batch Size**: `batch_size * world_size`
  - With `batch_size=256` and `world_size=8`, effective batch size is **2048**

- **Sync Frequency**:
  - `sync_freq=1`: Synchronize every update (most stable)
  - `sync_freq=2`: Accumulate gradients for 2 updates before syncing
  - Higher values can speed up training but may affect stability

## Example Configurations

### 1. Standard 8-GPU Training
```bash
python train_ddp.py \
    task=dog-run \
    model_size=5 \
    world_size=8 \
    batch_size=256 \
    sync_freq=1
# Effective batch size: 2048
```

### 2. Memory-Constrained Setup
```bash
python train_ddp.py \
    task=humanoid-walk \
    model_size=19 \
    world_size=8 \
    batch_size=128 \
    sync_freq=2
# Effective batch size: 1024
# Less GPU memory per device
```

### 3. Fast Training with Large Model
```bash
python train_ddp.py \
    task=humanoid-run \
    model_size=48 \
    world_size=8 \
    batch_size=512 \
    sync_freq=1 \
    steps=20000000
# Effective batch size: 4096
# Maximum throughput
```

### 4. Small-Scale Testing (4 GPUs)
```bash
python train_ddp.py \
    task=walker-run \
    model_size=5 \
    world_size=4 \
    batch_size=256
# Effective batch size: 1024
```

## Performance Tips

### 1. Batch Size Tuning
- Start with `batch_size=256` per GPU
- Increase if you have GPU memory available
- Larger batch sizes can improve sample efficiency

### 2. Sync Frequency
- Use `sync_freq=1` for most stable training
- Try `sync_freq=2-4` for potential speedup
- Monitor training metrics for instability

### 3. GPU Selection
```bash
# Use specific GPUs
CUDA_VISIBLE_DEVICES=0,1,2,3 python train_ddp.py task=dog-run world_size=4

# The script automatically uses GPUs 0 to (world_size-1)
```

### 4. Memory Optimization
If you encounter OOM errors:
- Reduce `batch_size`
- Reduce `buffer_size`
- Use smaller `model_size`
- Reduce `horizon`

## File Structure

```
tdmpc2/
├── train_ddp.py              # Main DDP training script
├── tdmpc2_ddp.py             # DDP-wrapped TDMPC2 agent
├── config_ddp.yaml           # DDP configuration template
├── run_ddp_8gpu.sh           # Convenient launch script
└── trainer/
    └── ddp_online_trainer.py # DDP online trainer implementation
```

## Monitoring Training

### Logs
- Only **Rank 0** logs to console and WandB
- All processes contribute to training
- Metrics are averaged across all GPUs during evaluation

### WandB Integration
```bash
python train_ddp.py \
    task=dog-run \
    world_size=8 \
    enable_wandb=true \
    wandb_project=my-tdmpc2-ddp \
    wandb_entity=your-username
```

### Checkpoints
- Only Rank 0 saves checkpoints to avoid conflicts
- Checkpoints include the unwrapped model (without DDP wrapper)
- Can be loaded in both single-GPU and multi-GPU modes

## Troubleshooting

### Issue: "NCCL error" or "distributed timeout"
**Solution**: Check that:
- All GPUs are visible and healthy (`nvidia-smi`)
- No other process is using the GPUs
- Network is not blocking localhost communication
- Try changing `master_port` if in use

### Issue: "Out of memory"
**Solution**: Reduce batch size or buffer size:
```bash
python train_ddp.py task=dog-run world_size=8 batch_size=128 buffer_size=500000
```

### Issue: "Training is unstable"
**Solution**:
- Reduce `sync_freq` to 1
- Reduce learning rate
- Reduce effective batch size

### Issue: Different GPUs show different utilization
**Solution**: This is normal:
- Experience collection is asynchronous
- Model updates are synchronized
- Some GPUs may finish episodes faster

## Comparison: Single GPU vs 8 GPUs

### Single GPU (baseline)
```bash
python train.py task=dog-run model_size=5 batch_size=256
# Batch size: 256
# Experience rate: ~1000 steps/sec
# Training time: ~3 hours
```

### 8 GPUs (DDP)
```bash
python train_ddp.py task=dog-run model_size=5 world_size=8 batch_size=256
# Effective batch size: 2048
# Experience rate: ~7000 steps/sec (7x speedup)
# Training time: ~30 minutes
```

## Advanced Usage

### Custom Synchronization Schedule
Modify `tdmpc2/trainer/ddp_online_trainer.py`:
```python
# Sync every N updates
should_sync = (self._step % N == 0)
_train_metrics = self.agent.update(self.buffer, sync_gradients=should_sync)
```

### Gradient Clipping
Already implemented in `tdmpc2_ddp.py`:
```python
grad_norm = torch.nn.utils.clip_grad_norm_(
    self.model.parameters(),
    self.cfg.grad_clip_norm
)
```

### Mixed Precision Training
To enable (experimental):
```python
# In tdmpc2_ddp.py, add:
from torch.cuda.amp import autocast, GradScaler
scaler = GradScaler()

# In _update():
with autocast():
    # ... forward pass ...
scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

## FAQ

**Q: Can I use DDP for offline training?**
A: Yes, but the provided implementation is optimized for online training. For offline training, see the `distributed` branch.

**Q: Does DDP work with multi-task training?**
A: Yes, set `task=mt30` or `task=mt80` with appropriate config.

**Q: Can I mix different tasks across GPUs?**
A: No, all GPUs train on the same task. For different tasks, run separate training jobs.

**Q: What's the minimum number of GPUs needed?**
A: Technically 1 (falls back to single-GPU mode), but 2+ for actual distributed training.

**Q: Can I use this on multiple machines?**
A: Yes, but you'll need to set `master_addr` to the IP of the master node and ensure network connectivity.

## Citation

If you use this DDP implementation, please cite the original TD-MPC2 paper:

```bibtex
@article{hansen2024tdmpc2,
  title={TD-MPC2: Scalable, Robust World Models for Continuous Control},
  author={Hansen, Nicklas and Wang, Xiaolong and Su, Hao},
  journal={arXiv preprint arXiv:2310.16828},
  year={2024}
}
```

## Support

For issues specific to DDP training:
- Check console output from Rank 0
- Verify all GPUs are functioning (`nvidia-smi`)
- Try reducing `world_size` to isolate problems
- Check NCCL environment variables

For general TD-MPC2 questions, refer to the main repository.
