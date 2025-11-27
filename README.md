# TD-MPC2 with Speculative Execution and Learned Corrector

This repository extends the open-source TD-MPC/TD-MPC2 control stack with speculative multi-step execution and a distillation-based corrector that imitates TD-MPC2 replanning. The goal is to accelerate inference by executing several planned actions before replanning while preserving robustness through a lightweight corrector trained from TD-MPC2 rollouts.

The codebase is intended for research on speeding up model-based control with minimal performance loss. It builds on the official TD-MPC/TD-MPC2 releases but is **not** an official implementation of those projects.

-----

## Features
- Single-task TD-MPC2-style training for continuous control tasks (Hydra configuration in `tdmpc2/config.yaml`).
- Speculative multi-step execution of TD-MPC2 plans to reduce replanning frequency.
- Learned corrector trained by distillation from a TD-MPC2 teacher to adjust speculative actions when real states deviate from predictions.
- End-to-end scripts for training the TD-MPC2 teacher, collecting distillation data, training the corrector, and evaluating speculative execution at different horizons.

-----

## Repository layout
- `tdmpc2/tdmpc2/` – TD-MPC2-style agent, speculative execution utilities, corrector implementations, and Hydra configs.
- `tdmpc2/scripts/` – Command-line entry points for corrector data collection and speculative-execution evaluation.
- `tdmpc2/docker/` – Example conda environment (`environment.yaml`) and Dockerfile for running MuJoCo-based tasks.
- `logs/` (created at runtime) – Default location for training/evaluation logs and checkpoints.

-----

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

-----

## Using official pretrained TD-MPC2 teachers
You can skip training the teacher from scratch by downloading the official checkpoints (all sizes except the ~1M model):

```bash
cd tdmpc2
python scripts/download_tdmpc2_models.py \
  --output_dir tdmpc2_pretrained
```

- By default the downloader skips the smallest model; pass `--include_smallest` if you explicitly want it.
- Checkpoints are saved as `tdmpc2_pretrained/tdmpc2_<size>.pt` (e.g., `tdmpc2_5m.pt`, `tdmpc2_19m.pt`, `tdmpc2_48m.pt`, `tdmpc2_317m.pt`).
- You can also point `--manifest` to a JSON mapping of `{ "5m": "https://...pt", ... }` to override URLs.

-----

## Training the TD-MPC2 Teacher
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

-----

## Collecting corrector training data from pretrained teachers
Use the downloaded pretrained TD-MPC2 checkpoints as frozen teachers. The script can target a single model size or iterate over all downloaded sizes (excluding ~1M by default):

**Single size example (5M teacher)**
```bash
cd tdmpc2
python scripts/collect_corrector_data.py \
  --task humanoid-run \
  --model_size 5m \
  --model_dir tdmpc2_pretrained \
  --episodes 20 \
  --plan_horizon 3 \
  --history_len 4 \
  --output data/corrector_data_5m.pt \
  --device cuda
```

**All available pretrained sizes (auto-detected in `model_dir`)**
```bash
python scripts/collect_corrector_data.py \
  --task humanoid-run \
  --all_model_sizes \
  --model_dir tdmpc2_pretrained \
  --episodes 20 \
  --plan_horizon 3 \
  --history_len 4 \
  --output data/corrector_data.pt
```

- The default output name is automatically expanded to `data/corrector_data_<size>.pt` when `--all_model_sizes` is used.
- Each dataset stores `z_real`, `z_pred`, `a_plan`, `a_teacher`, `distance`, and `history_feats` so both the two-tower and temporal correctors can train from the same file.

-----

## Training the corrector (per pretrained size)
Train both corrector architectures on the collected buffers. The trainer automatically picks the right dataset naming pattern and saves per-size checkpoints.

**Train both correctors for a single pretrained size**
```bash
cd tdmpc2
python tdmpc2/train_corrector.py \
  --model_size 5m \
  --data_dir data \
  --corrector_type both \
  --epochs 20 \
  --batch_size 256 \
  --history_len 4 \
  --device cuda
```
This produces `correctors/corrector_5m_two_tower.pth` and `correctors/corrector_5m_temporal.pth`.

**Train across every available dataset (matching all downloaded pretrained sizes)**
```bash
python tdmpc2/train_corrector.py \
  --model_size all \
  --data_dir data \
  --corrector_type both \
  --epochs 20 \
  --batch_size 256 \
  --history_len 4
```
- Use `--corrector_type two_tower` or `--corrector_type temporal` to limit training to a single architecture.
- Pass `--data <path>` to target a custom dataset file instead of per-size discovery.

-----

## Evaluating speculative execution (baseline vs. correctors)
`scripts/eval_corrector.py` now evaluates baseline TD-MPC2 replanning, open-loop execution (2- and 3-step), and both correctors for each pretrained size. Results are aggregated into a CSV for plotting.

**Evaluate a single pretrained size (reads checkpoints from `model_dir` and correctors from `corrector_dir`)**
```bash
cd tdmpc2
python scripts/eval_corrector.py \
  --task humanoid-run \
  --model_size 5m \
  --model_dir tdmpc2_pretrained \
  --corrector_dir correctors \
  --episodes 10 \
  --spec_plan_horizon 3 \
  --device cuda \
  --results_csv results/corrector_eval/summary.csv
```

**Evaluate all downloaded pretrained sizes**
```bash
python scripts/eval_corrector.py \
  --task humanoid-run \
  --all_model_sizes \
  --model_dir tdmpc2_pretrained \
  --corrector_dir correctors \
  --episodes 10 \
  --spec_plan_horizon 3 \
  --results_csv results/corrector_eval/summary.csv
```

- Per-run JSON/CSV metrics are written under `results/corrector_eval/`, and an aggregated summary CSV is saved to `--results_csv`.
- Use `scripts/plot_corrector_eval.py --results_csv results/corrector_eval/summary.csv --output_dir results/corrector_eval/plots` to generate horizon/model-size/improvement plots.

-----

## Distributed Multi-GPU Training (DDP)
Use the provided DDP tooling to scale TD-MPC2 training across multiple GPUs:

- Quick-start shell script (8 GPUs by default):
  ```bash
  cd tdmpc2
  ./run_ddp_8gpu.sh dog-run 5
  ```
- Direct Python invocation with custom settings:
  ```bash
  cd tdmpc2
  python train_ddp.py task=humanoid-run model_size=19 world_size=4 sync_freq=2 batch_size=256
  ```
- Config-driven launch:
  ```bash
  cd tdmpc2
  python train_ddp.py --config-name=config_ddp
  ```

  See `tdmpc2/DDP_TRAINING_README.md` for detailed guidance on parameters like `world_size`, `sync_freq`, and troubleshooting tips. Rank 0 handles logging and checkpoints; checkpoints remain loadable for single- or multi-GPU use.
  
-----

## Reproducibility and Logging
- Set `seed=<int>` (Hydra for training/evaluation; CLI flag for scripts) to control randomness.
- Training/evaluation logs, videos, and checkpoints default to `logs/<task>/<seed>/<exp_name>/`.
- For stable comparisons, run multiple seeds and report average returns.

-----

## Citing & Acknowledgements
This repository builds on TD-MPC and TD-MPC2 but is an independent extension with speculative execution and a learned corrector.
- TD-MPC code: https://github.com/nicklashansen/tdmpc  |  Paper: “Temporal Difference Learning for Model Predictive Control” (Hansen et al., ICML 2022), arXiv:2203.04955.
- TD-MPC2 code: https://github.com/nicklashansen/tdmpc2  |  Paper: “TD-MPC2: Scalable, Robust World Models for Continuous Control”, arXiv:2310.16828.

-----

## License
This project is licensed under the terms of the existing `LICENSE` file (MIT).
