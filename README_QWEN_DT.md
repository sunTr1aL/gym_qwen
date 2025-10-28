
# Qwen3-Style Decision Transformer (for MuJoCo Humanoid)

This adds a **Qwen3-style Transformer backbone** to the original Decision Transformer codebase you uploaded,
without changing the original folder layout. New files:

```
gym/decision_transformer/models/qwen3_transformer.py
gym/decision_transformer/models/decision_transformer_qwen3.py
gym/experiment_qwen.py
gym/train_humanoid_qwen_dt.py
gym/requirements_qwen.txt
```

## Key Differences vs GPT-2 DT
- **RoPE** positional encoding (long-context friendly)
- **RMSNorm + Pre-Norm**
- **SwiGLU** FFN (wider `mlp_ratio`, defaults to 5.4×)
- **Grouped Query Attention (GQA)** with `n_kv_head`
- No token embeddings inside the Transformer; we feed `inputs_embeds` directly like the original DT

## Quick Start

1. (Optional) Create env from `conda_env.yml` in your repo or install minimal requirements:
   ```bash
   pip install -r gym/requirements_qwen.txt
   ```

2. Download D4RL Humanoid datasets (if not already):
   ```bash
   python gym/data/download_d4rl_datasets.py --env humanoid --out gym/data
   ```

3. Train (0.6B-style config; needs strong GPU memory):
   ```bash
   cd gym
   python train_humanoid_qwen_dt.py
   ```

   For a smaller sanity check:
   ```bash
   python experiment_qwen.py --env humanoid-medium-v2 --dataset medium \
     --embed_dim 768 --n_layer 12 --n_head 12 --n_kv_head 4 --mlp_ratio 4.0 \
     --batch_size 64 --K 20 --max_iters 2 --num_steps_per_iter 1000
   ```

## Notes
- **Action head** uses `tanh` to fit MuJoCo action range `[-1, 1]`.
- The trainer and evaluation path reuse your original `training/` and `evaluation/` modules.
- You can switch to other D4RL tasks by changing `--env` / `--dataset` arguments.
- This implementation focuses on *training-time* GQA and RoPE; KV-cache optimizations for inference are not included (not required for DT training).

## Suggested Large Config (≈0.6B)
- `embed_dim=2048`, `n_layer=24`, `n_head=16`, `n_kv_head=8`, `mlp_ratio=5.4`
- Sequence length `K=20` is common in DT; adjust based on memory.
- Mixed precision (`torch.cuda.amp`) can be enabled if desired in the trainer.

Good luck and happy training!
