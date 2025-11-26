# TD-MPC2 with Speculative Execution and Learned Corrector

This repository extends the open-source TD-MPC/TD-MPC2 control stack with speculative multi-step execution and a distillation-based corrector that imitates TD-MPC2 replanning. The goal is to accelerate inference by executing several planned actions before replanning while preserving robustness through a lightweight corrector trained from TD-MPC2 rollouts.

The codebase is intended for research on speeding up model-based control with minimal performance loss. It builds on the official TD-MPC/TD-MPC2 releases but is **not** an official implementation of those projects.

## Features
- Single-task TD-MPC2-style training for continuous control tasks (Hydra configuration in `tdmpc2/config.yaml`).
- Speculative multi-step execution of TD-MPC2 plans to reduce replanning frequency.
- Learned corrector trained by distillation from a TD-MPC2 teacher to adjust speculative actions when real states deviate from predictions.
- End-to-end scripts for training the TD-MPC2 teacher, collecting distillation data, training the corrector, and evaluating speculative execution at different horizons.

## Repository layout
- `tdmpc2/tdmpc2/` – TD-MPC2-style agent, speculative execution utilities, corrector implementations, and Hydra configs.
- `tdmpc2/scripts/` – Command-line entry points for corrector data collection and speculative-execution evaluation.
- `tdmpc2/docker/` – Example conda environment (`environment.yaml`) and Dockerfile for running MuJoCo-based tasks.
- `logs/` (created at runtime) – Default location for training/evaluation logs and checkpoints.

## Installation
1. Use Python 3.9+.
2. Create an environment (conda example):
   ```bash
   conda env create -f tdmpc2/docker/environment.yaml
   conda activate tdmpc2
   ```
3. Or with `venv` (install dependencies matching `docker/environment.yaml`):
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install --upgrade pip
   # Install packages listed in tdmpc2/docker/environment.yaml (dm-control, gymnasium, hydra-core, mujoco, torch, etc.)
   ```
4. Ensure MuJoCo and required control suites are available (e.g., `dm-control`, `mujoco` Python package). Set `MUJOCO_GL=egl` if running headless.

## 5. Training the TD-MPC2 Teacher
Train the single-task TD-MPC2-style agent with Hydra (defaults in `tdmpc2/config.yaml`). Example:
```bash
cd tdmpc2
python tdmpc2/train.py \
  task=humanoid-run \
  model_size=5 \
  steps=1000000 \
  seed=1 \
  device=cuda
```
- `task`/`env` selects the Gym/DMControl-style environment name.
- Checkpoints and logs are stored under `logs/<task>/<seed>/<exp_name>/`.
- Use `device=cpu` to run without CUDA (slower).

## 6. Collecting Corrector Training Data
Generate distillation data from the TD-MPC2 teacher for the learned corrector:
```bash
cd tdmpc2
python scripts/collect_corrector_data.py \
  --task humanoid-run \
  --checkpoint checkpoints/tdmpc_teacher.pt \
  --episodes 20 \
  --output data/corrector_data.pt \
  --device cuda \
  --plan_horizon 3 \
  --teacher_interval 1
```
The script runs the TD-MPC2 teacher in evaluation mode, logs planned actions and TD-MPC2 replans when real states deviate from predictions, and saves tensors (`z_real`, `z_pred`, `a_plan`, `a_teacher`, `distance`, optional history features) for offline corrector training.

## 7. Training the Corrector (Distillation)
Train the learned corrector on the collected dataset:
```bash
cd tdmpc2
# Two-tower corrector
python tdmpc2/train_corrector.py \
  --data data/corrector_data.pt \
  --tdmpc_ckpt checkpoints/tdmpc_teacher.pt \
  --save_path checkpoints/corrector_two_tower.pth \
  --corrector_type two_tower \
  --epochs 20 \
  --batch_size 256

# Temporal transformer corrector
python tdmpc2/train_corrector.py \
  --data data/corrector_data.pt \
  --tdmpc_ckpt checkpoints/tdmpc_teacher.pt \
  --save_path checkpoints/corrector_temporal.pth \
  --corrector_type temporal \
  --history_len 4 \
  --epochs 20
```
The corrector is distilled to imitate TD-MPC2 replanning behavior so speculative actions can be adjusted when real trajectories diverge from planned ones.

## 8. Evaluating Speculative Execution
Use `scripts/eval_corrector.py` to compare baseline TD-MPC2 and speculative execution variants.

1. **Baseline TD-MPC2 (no speculation)**
   ```bash
   cd tdmpc2
   python scripts/eval_corrector.py \
     --task humanoid-run \
     --tdmpc_checkpoint checkpoints/tdmpc_teacher.pt \
     --mode baseline \
     --episodes 10 \
     --device cuda
   ```

2. **Speculative execution without corrector (3-step horizon)**
   ```bash
   python scripts/eval_corrector.py \
     --task humanoid-run \
     --tdmpc_checkpoint checkpoints/tdmpc_teacher.pt \
     --mode naive3 \
     --spec_plan_horizon 3 \
     --spec_exec_horizon 3 \
     --spec_mismatch_threshold 0.5
   ```

3. **Speculative execution with corrector (3-step horizon)**
   ```bash
   python scripts/eval_corrector.py \
     --task humanoid-run \
     --tdmpc_checkpoint checkpoints/tdmpc_teacher.pt \
     --corrector_checkpoint checkpoints/corrector_two_tower.pth \
     --corrector_type two_tower \
     --mode spec_corrector \
     --spec_plan_horizon 3 \
     --spec_exec_horizon 3 \
     --spec_mismatch_threshold 0.5
   ```

4. **Extended speculative execution with corrector (e.g., 6-step horizon)**
   ```bash
   python scripts/eval_corrector.py \
     --task humanoid-run \
     --tdmpc_checkpoint checkpoints/tdmpc_teacher.pt \
     --corrector_checkpoint checkpoints/corrector_temporal.pth \
     --corrector_type temporal \
     --mode spec6_corrector \
     --spec_plan_horizon 6 \
     --spec_exec_horizon 6 \
     --spec_mismatch_threshold 0.5
   ```

Metrics are printed to stdout; optionally save JSON metrics with `--output_metrics_path <file>`.

## 9. Reproducibility and Logging
- Set `seed=<int>` (Hydra for training/evaluation; CLI flag for scripts) to control randomness.
- Training/evaluation logs, videos, and checkpoints default to `logs/<task>/<seed>/<exp_name>/`.
- For stable comparisons, run multiple seeds and report average returns.

## 10. Citing & Acknowledgements
This repository builds on TD-MPC and TD-MPC2 but is an independent extension with speculative execution and a learned corrector.
- TD-MPC code: https://github.com/nicklashansen/tdmpc  |  Paper: “Temporal Difference Learning for Model Predictive Control” (Hansen et al., ICML 2022), arXiv:2203.04955.
- TD-MPC2 code: https://github.com/nicklashansen/tdmpc2  |  Paper: “TD-MPC2: Scalable, Robust World Models for Continuous Control”, arXiv:2310.16828.

## 11. License
This project is licensed under the terms of the existing `LICENSE` file (MIT).
