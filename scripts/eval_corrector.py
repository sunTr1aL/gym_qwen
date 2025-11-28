#!/usr/bin/env python3
"""Evaluate TD-MPC2 open-loop execution with and without corrective models."""

from __future__ import annotations

import argparse
import csv
import datetime
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "tdmpc2"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from common.seed import set_seed
from envs import make_env
from tdmpc2 import TDMPC2
from utils_ckpt import list_pretrained_checkpoints, load_pretrained_tdmpc2

EVAL_VARIANTS = [
    {"name": "baseline_replan", "exec_horizon": 1, "corrector_type": None},
    {"name": "open_loop_2", "exec_horizon": 2, "corrector_type": None},
    {"name": "open_loop_3", "exec_horizon": 3, "corrector_type": None},
    {"name": "corrected_two_tower_2", "exec_horizon": 2, "corrector_type": "two_tower"},
    {"name": "corrected_temporal_2", "exec_horizon": 2, "corrector_type": "temporal"},
    {"name": "corrected_two_tower_3", "exec_horizon": 3, "corrector_type": "two_tower"},
    {"name": "corrected_temporal_3", "exec_horizon": 3, "corrector_type": "temporal"},
]


def _resolve_models(args: argparse.Namespace) -> Iterable[tuple[str, Dict[str, str]]]:
    ckpts = list_pretrained_checkpoints(
        args.checkpoint_dir, model_size_filter=args.model_size
    )
    if args.all_models or args.all_model_sizes or not args.model_id:
        if not ckpts:
            raise ValueError(f"No pretrained checkpoints found in {args.checkpoint_dir}")
        return ckpts.items()
    if args.model_id not in ckpts:
        raise ValueError(
            f"Model id '{args.model_id}' not found in {args.checkpoint_dir}. Available: {list(ckpts.keys())}"
        )
    return [(args.model_id, ckpts[args.model_id])]


def _corrector_ckpt_for(model_id: str, corr_type: Optional[str], args: argparse.Namespace) -> Optional[str]:
    if corr_type is None:
        return None
    if args.corrector_checkpoint:
        return args.corrector_checkpoint
    ckpt_path = Path(args.corrector_dir) / f"corrector_{model_id}_{corr_type}.pth"
    return str(ckpt_path)


def _build_agent(model_id: str, ckpt_path: str, variant: Dict[str, Any], args: argparse.Namespace):
    corr_type = variant["corrector_type"]
    exec_h = variant["exec_horizon"]
    corrector_ckpt = _corrector_ckpt_for(model_id, corr_type, args)
    agent, cfg = load_pretrained_tdmpc2(
        checkpoint_path=ckpt_path,
        device=args.device,
        model_id=model_id,
        task=args.task,
        obs_type=args.obs_type,
    )
    return agent, cfg, {}, corrector_ckpt


def run_rollout(agent: TDMPC2, env, episodes: int, max_steps: int) -> Dict[str, List[float]]:
    returns: List[float] = []
    lengths: List[int] = []
    replan_counts: List[int] = []
    corrector_steps: List[int] = []
    total_steps = 0
    start = time.time()
    for ep in range(episodes):
        reset_out = env.reset()
        obs = reset_out[0] if isinstance(reset_out, tuple) else reset_out
        done, ep_steps = False, 0
        ep_return = 0.0
        local_corrector_steps = 0
        while not done and (max_steps <= 0 or ep_steps < max_steps):
            action_out = agent.act(
                torch.as_tensor(obs, device=agent.device, dtype=torch.float32),
                t0=ep_steps == 0,
                eval_mode=True,
                return_info=True,
            )
            if isinstance(action_out, tuple) and len(action_out) == 2:
                action, info = action_out
            else:
                action, info = action_out, {}
            next_obs, reward, done, _ = env.step(action.cpu().numpy())
            obs = next_obs[0] if isinstance(next_obs, tuple) else next_obs
            ep_return += float(reward)
            ep_steps += 1
            total_steps += 1
            if isinstance(info, dict) and info.get("used_corrector"):
                local_corrector_steps += 1
        returns.append(ep_return)
        lengths.append(ep_steps)
        replan_counts.append(int(getattr(agent, "episode_replans", 0)))
        corrector_steps.append(int(getattr(agent, "episode_corrector_steps", local_corrector_steps)))
        print(
            f"Episode {ep+1}: return={ep_return:.2f}, steps={ep_steps}, replans={replan_counts[-1]}, "
            f"corrector_steps={corrector_steps[-1]}"
        )
    elapsed = time.time() - start
    steps_per_sec = total_steps / max(elapsed, 1e-6)
    print(f"[throughput] {steps_per_sec:.1f} env steps/sec across {episodes} episodes")
    return {
        "returns": returns,
        "lengths": lengths,
        "replans": replan_counts,
        "corrector_steps": corrector_steps,
    }


def summarize(metrics: Dict[str, List[float]]) -> Dict[str, float]:
    returns = np.array(metrics["returns"])
    lengths = np.array(metrics["lengths"])
    replans = np.array(metrics["replans"])
    corr_steps = np.array(metrics["corrector_steps"])
    return {
        "mean_return": float(returns.mean()),
        "median_return": float(np.median(returns)),
        "std_return": float(returns.std()),
        "p5_return": float(np.percentile(returns, 5.0)),
        "mean_length": float(lengths.mean()),
        "mean_replans": float(replans.mean()),
        "mean_corrector_steps": float(corr_steps.mean()),
    }


def _save_run_files(
    meta: Dict[str, Any],
    metrics: Dict[str, List[float]],
    summary: Dict[str, float],
    args: argparse.Namespace,
) -> None:
    ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    run_id = f"{meta['task']}_{meta['variant']}_{meta['model_id']}_{meta['corrector_type'] or 'none'}_seed{meta['seed']}"
    base = Path(args.results_dir) / f"{run_id}_{ts}"
    base.parent.mkdir(parents=True, exist_ok=True)
    with open(str(base) + "_eval.json", "w", encoding="utf-8") as f:
        json.dump({"meta": meta, "summary": summary, "metrics": metrics}, f, indent=2)
    with open(str(base) + "_eval.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["episode", "return", "length", "num_replans", "num_corrector_steps"])
        for i, (r, l, nr, nc) in enumerate(
            zip(metrics["returns"], metrics["lengths"], metrics["replans"], metrics["corrector_steps"])
        ):
            writer.writerow([i, r, l, nr, nc])


def main_worker(rank: int, world_size: int, args: argparse.Namespace) -> None:
    del world_size  # evaluation runs in a single process

    set_seed(args.seed)
    results: List[Dict[str, Any]] = []

    for model_id, info in _resolve_models(args):
        ckpt_path = info["path"]
        model_name = info.get("model_name", "")
        model_size = info.get("model_size", "")
        for variant in EVAL_VARIANTS:
            corr_type = variant["corrector_type"]
            agent, cfg, _, corrector_ckpt = _build_agent(model_id, ckpt_path, variant, args)
            if torch.cuda.is_available() and torch.cuda.device_count() > 1:
                agent.model = nn.DataParallel(agent.model)
                if getattr(agent, "corrector", None) is not None:
                    agent.corrector = nn.DataParallel(agent.corrector)
            if not hasattr(cfg, "obs") or str(cfg.obs).lower() not in {"state", "rgb"}:
                cfg.obs = args.obs_type.lower()
            else:
                cfg.obs = str(cfg.obs).lower()
            cfg.obs_type = str(getattr(cfg, "obs_type", cfg.obs)).lower()
            if cfg.obs_type not in {"state", "rgb"}:
                cfg.obs_type = cfg.obs
            env = make_env(cfg)
            metrics = run_rollout(agent, env, episodes=args.episodes, max_steps=args.max_steps)
            summary = summarize(metrics)
            meta = {
                "task": args.task or cfg.task,
                "variant": variant["name"],
                "model_id": model_id,
                "model_name": model_name,
                "model_size": model_size,
                "corrector_type": corr_type,
                "exec_horizon": variant["exec_horizon"],
                "episodes": args.episodes,
                "seed": cfg.seed,
                "tdmpc_checkpoint": cfg.checkpoint,
                "corrector_checkpoint": corrector_ckpt,
            }
            print("Summary:", summary)
            _save_run_files(meta, metrics, summary, args)
            record = {**meta, **summary}
            results.append(record)

    results_csv = Path(args.results_csv)
    results_csv.parent.mkdir(parents=True, exist_ok=True)
    if results:
        with open(results_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=list(results[0].keys()),
            )
            writer.writeheader()
            writer.writerows(results)
        print(f"Saved aggregated results to {results_csv}")
    else:
        print("No results produced; check configuration.")

    if results and args.output_plot:
        try:
            import matplotlib.pyplot as plt
            import pandas as pd

            df = pd.DataFrame(results)
            df["corrector_label"] = df["corrector_type"].fillna("none")
            grouped = df.groupby(["corrector_label", "exec_horizon"])["mean_return"].mean().unstack(fill_value=0)
            horizons = grouped.columns.tolist()
            plt.figure(figsize=(8, 5))
            for corr_type, row in grouped.iterrows():
                plt.plot(horizons, row.values, marker="o", label=corr_type)
            plt.xlabel("Execution horizon (steps)")
            plt.ylabel("Mean return (avg across models)")
            plt.title("Performance vs execution horizon")
            plt.grid(True, linestyle="--", alpha=0.5)
            plt.legend(title="Corrector")
            Path(args.output_plot).parent.mkdir(parents=True, exist_ok=True)
            plt.tight_layout()
            plt.savefig(args.output_plot, dpi=200)
            print(f"Saved aggregate plot to {args.output_plot}")
        except Exception as exc:  # pragma: no cover - plotting optional
            print(f"Failed to generate aggregate plot: {exc}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", "--env", dest="task", type=str, help="Task name / env id", required=False)
    parser.add_argument("--model_id", type=str, default=None, help="Checkpoint stem to evaluate (e.g., tdmpc2_19m)")
    parser.add_argument("--model_size", type=str, default=None, help="Filter checkpoints by size token (e.g., 5m)")
    parser.add_argument("--all_models", action="store_true", help="Evaluate every checkpoint discovered in checkpoint_dir")
    parser.add_argument("--all_model_sizes", action="store_true", help="Alias for --all_models")
    parser.add_argument("--checkpoint_dir", type=str, default="tdmpc2_pretrained", help="Directory with pretrained models")
    parser.add_argument("--corrector_dir", type=str, default="correctors", help="Directory containing trained correctors")
    parser.add_argument("--corrector_checkpoint", type=str, default=None, help="Override corrector checkpoint path")
    parser.add_argument("--spec_plan_horizon", type=int, default=3)
    parser.add_argument("--spec_exec_horizon", type=int, default=3, help="Unused placeholder for backward compatibility")
    parser.add_argument("--spec_mismatch_threshold", type=float, default=0.5)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--max_steps", type=int, default=-1)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument(
        "--results_dir",
        type=str,
        default="results/corrector_eval",
        help="Directory to save per-run evaluation metrics (JSON/CSV).",
    )
    parser.add_argument("--results_csv", type=str, default="results/corrector_eval/summary.csv")
    parser.add_argument("--output_csv", dest="results_csv", type=str, help="Alias for --results_csv")
    parser.add_argument(
        "--output_plot",
        type=str,
        default=None,
        help="Optional path to save a quick aggregate horizon plot (uses aggregated CSV).",
    )
    parser.add_argument("--results_csv", type=str, default="results/corrector_eval/summary.csv")
    parser.add_argument(
        "--obs_type",
        type=str,
        default="state",
        choices=["state", "rgb"],
        help="Observation type for dmcontrol envs.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    _args = parse_args()
    if _args.all_model_sizes:
        _args.all_models = True
    main_worker(rank=0, world_size=1, args=_args)
