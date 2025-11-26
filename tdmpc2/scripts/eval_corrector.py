#!/usr/bin/env python3
"""Evaluate TD-MPC2 with speculative execution and correctors."""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

import torch
from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from common.parser import parse_cfg  # noqa: E402
from common.seed import set_seed  # noqa: E402
from envs import make_env  # noqa: E402
from tdmpc2 import TDMPC2  # noqa: E402


def build_cfg(args: argparse.Namespace) -> Any:
    cfg_path = Path(args.config) if args.config else REPO_ROOT / "tdmpc2" / "config.yaml"
    cfg = OmegaConf.load(cfg_path)
    cfg.task = args.task or cfg.get("task")
    cfg.device = args.device
    cfg.seed = args.seed
    cfg.eval_episodes = args.episodes
    cfg.checkpoint = args.tdmpc_checkpoint
    cfg.spec_enabled = args.mode != "baseline"
    cfg.spec_plan_horizon = args.spec_plan_horizon
    cfg.spec_exec_horizon = args.spec_exec_horizon
    cfg.spec_mismatch_threshold = args.spec_mismatch_threshold
    cfg.use_corrector = args.mode in {"spec_corrector", "spec6_corrector"}
    cfg.corrector_ckpt = args.corrector_checkpoint
    cfg.corrector_type = args.corrector_type
    cfg.speculate = False
    cfg = parse_cfg(cfg)
    return cfg


def run_rollout(agent: TDMPC2, env, episodes: int, max_steps: int) -> Dict[str, List[float]]:
    returns: List[float] = []
    lengths: List[int] = []
    replan_counts: List[int] = []
    for ep in range(episodes):
        obs, done, ep_steps = env.reset(), False, 0
        ep_return = 0.0
        replan_counter = 0
        while not done and (max_steps <= 0 or ep_steps < max_steps):
            action = agent.act(torch.as_tensor(obs, device=agent.device, dtype=torch.float32), t0=ep_steps == 0, eval_mode=True)
            if isinstance(action, tuple):
                action = action[0]
            next_obs, reward, done, _ = env.step(action.cpu().numpy())
            obs = next_obs
            ep_return += float(reward)
            ep_steps += 1
            if getattr(agent, "steps_until_replan", 0) == agent.spec_exec_horizon:
                replan_counter += 1
        returns.append(ep_return)
        lengths.append(ep_steps)
        replan_counts.append(replan_counter)
        print(f"Episode {ep+1}: return={ep_return:.2f}, steps={ep_steps}")
    return {"returns": returns, "lengths": lengths, "replans": replan_counts}


def summarize(metrics: Dict[str, List[float]]) -> Dict[str, float]:
    import numpy as np

    returns = np.array(metrics["returns"])
    return {
        "mean_return": float(returns.mean()),
        "median_return": float(np.median(returns)),
        "p5_return": float(np.percentile(returns, 5)),
        "avg_length": float(np.mean(metrics["lengths"])),
        "avg_replans": float(np.mean(metrics["replans"])),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", "--env", dest="task", type=str, help="Task name / env id", required=False)
    parser.add_argument("--tdmpc_checkpoint", type=str, required=True, help="Path to TD-MPC2 checkpoint")
    parser.add_argument("--corrector_checkpoint", type=str, default=None, help="Path to trained corrector")
    parser.add_argument("--corrector_type", type=str, default="two_tower", choices=["two_tower", "temporal"])
    parser.add_argument("--mode", type=str, default="baseline", choices=["baseline", "naive3", "spec_corrector", "spec6_corrector"])
    parser.add_argument("--spec_plan_horizon", type=int, default=3)
    parser.add_argument("--spec_exec_horizon", type=int, default=3)
    parser.add_argument("--spec_mismatch_threshold", type=float, default=0.5)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--max_steps", type=int, default=-1)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--output_metrics_path", type=str, default=None)
    args = parser.parse_args()

    set_seed(args.seed)
    cfg = build_cfg(args)
    env = make_env(cfg)
    agent = TDMPC2(cfg)
    agent.load(args.tdmpc_checkpoint)
    agent.eval()

    metrics = run_rollout(agent, env, episodes=args.episodes, max_steps=args.max_steps)
    summary = summarize(metrics)
    print("Summary:", summary)
    if args.output_metrics_path:
        os.makedirs(os.path.dirname(args.output_metrics_path), exist_ok=True)
        with open(args.output_metrics_path, "w", encoding="utf-8") as f:
            json.dump({"metrics": metrics, "summary": summary}, f, indent=2)


if __name__ == "__main__":
    main()
