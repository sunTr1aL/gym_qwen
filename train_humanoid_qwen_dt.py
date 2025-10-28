
"""
Quick launcher for Qwen3-0.6B style Decision Transformer on MuJoCo Humanoid (D4RL).
Usage:
    python train_humanoid_qwen_dt.py --env humanoid-expert-v2 --dataset expert --log_to_wandb
"""
import sys
import subprocess

def main():
    cmd = [
        sys.executable, "experiment_qwen.py",
        "--env", "humanoid-expert-v2",
        "--dataset", "expert",
        "--model_type", "dt",
        "--K", "20",
        "--batch_size", "32",
        "--embed_dim", "2048",
        "--n_layer", "24",
        "--n_head", "16",
        "--n_kv_head", "8",
        "--mlp_ratio", "5.4",
        "--learning_rate", "1e-4",
        "--weight_decay", "0.1",
        "--warmup_steps", "10000",
        "--num_eval_episodes", "10",
        "--max_iters", "20",
        "--num_steps_per_iter", "10000",
        "--device", "cuda",
        "--scale", "1000.0",
        "--mode", "normal",
    ]
    print("Launching:", " ".join(cmd))
    raise SystemExit(subprocess.call(cmd))

if __name__ == "__main__":
    main()
