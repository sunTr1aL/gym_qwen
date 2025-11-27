#!/usr/bin/env python3
"""Collect distillation data for speculative correctors.

This script runs a trained TD-MPC2 agent in evaluation mode, plans short open-loop
sequences, and logs tuples for training residual correctors:

    (z_real, z_pred, a_plan, a_teacher, distance, history_feats)

The teacher action is obtained by replanning from the real next observation. History
features allow the temporal transformer corrector to reason over recent mismatches.
"""

import argparse
import os
import sys
import time
from collections import deque
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import torch
from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

DEFAULT_OUTPUT = "data/corrector_data.pt"

from common.parser import parse_cfg  # noqa: E402
from common.seed import set_seed  # noqa: E402
from envs import make_env  # noqa: E402
from tdmpc2 import TDMPC2  # noqa: E402
from tdmpc2.launch import launch, wrap_dataparallel
from tdmpc2.utils_ckpt import list_pretrained_checkpoints, load_pretrained_tdmpc2  # noqa: E402


def default_config_path() -> Path:
    return REPO_ROOT / "tdmpc2" / "config.yaml"


class CorrectorDataset:
    """Lightweight in-memory buffer for corrector samples."""

    def __init__(self) -> None:
        self.z_real: List[torch.Tensor] = []
        self.z_pred: List[torch.Tensor] = []
        self.a_plan: List[torch.Tensor] = []
        self.a_teacher: List[torch.Tensor] = []
        self.distance: List[float] = []
        self.history_feats: List[torch.Tensor] = []

    def add(
        self,
        z_real: torch.Tensor,
        z_pred: torch.Tensor,
        a_plan: torch.Tensor,
        a_teacher: torch.Tensor,
        distance: float,
        history_feats: torch.Tensor,
    ) -> None:
        self.z_real.append(z_real.detach().cpu())
        self.z_pred.append(z_pred.detach().cpu())
        self.a_plan.append(a_plan.detach().cpu())
        self.a_teacher.append(a_teacher.detach().cpu())
        self.distance.append(float(distance))
        self.history_feats.append(history_feats.detach().cpu())

    def __len__(self) -> int:  # pragma: no cover - trivial
        return len(self.z_real)

    def to_tensor_dict(self, device: str = "cpu") -> Dict[str, torch.Tensor]:
        target_device = torch.device(device)
        num = len(self)
        if num == 0:
            return {
                "z_real": torch.empty((0,), device=target_device),
                "z_pred": torch.empty((0,), device=target_device),
                "a_plan": torch.empty((0,), device=target_device),
                "a_teacher": torch.empty((0,), device=target_device),
                "distance": torch.empty((0,), device=target_device),
                "history_feats": torch.empty((0,), device=target_device),
            }
        return {
            "z_real": torch.stack(self.z_real).to(target_device),
            "z_pred": torch.stack(self.z_pred).to(target_device),
            "a_plan": torch.stack(self.a_plan).to(target_device),
            "a_teacher": torch.stack(self.a_teacher).to(target_device),
            "distance": torch.tensor(self.distance, device=target_device),
            "history_feats": torch.stack(self.history_feats).to(target_device),
        }


def build_cfg(args: argparse.Namespace) -> Any:
    cfg_path = Path(args.config) if args.config else default_config_path()
    cfg = OmegaConf.load(cfg_path)
    cfg.task = args.task or cfg.get("task")
    cfg.device = args.device
    cfg.seed = args.seed
    cfg.eval_episodes = args.episodes
    cfg.checkpoint = args.checkpoint
    cfg.spec_enabled = False  # teacher replans every step for data collection
    cfg.speculate = False
    cfg.use_corrector = False
    if args.max_steps is not None:
        cfg.episode_length = args.max_steps
    cfg = parse_cfg(cfg)
    return cfg


def pad_history(history: deque, target_len: int, feat_dim: int) -> torch.Tensor:
    missing = target_len - len(history)
    if missing > 0:
        zero_feat = torch.zeros(feat_dim)
        padded = [zero_feat for _ in range(missing)] + list(history)
    else:
        padded = list(history)[-target_len:]
    return torch.stack(padded, dim=0)


def collect_for_agent(agent: TDMPC2, cfg: Any, args: argparse.Namespace, output_path: str) -> None:
    buffer = CorrectorDataset()
    max_samples = args.max_samples if args.max_samples and args.max_samples > 0 else None
    history = deque(maxlen=args.history_len)
    feat_dim = 3 * cfg.latent_dim + cfg.action_dim
    env = make_env(cfg)

    task = getattr(env, "task", None) if hasattr(env, "task") else None
    if task is None:
        task = getattr(cfg, "task", None)

    print(f"Collecting corrector data for task {cfg.task} on device {cfg.device} -> {output_path}")
    episodes = 0
    start_time = time.time()
    total_steps = 0
    try:
        while episodes < args.episodes and (max_samples is None or len(buffer) < max_samples):
            obs, done, ep_steps = env.reset(), False, 0
            history.clear()
            while not done and (args.max_steps is None or ep_steps < args.max_steps):
                obs_tensor = torch.as_tensor(obs, device=agent.device, dtype=torch.float32)
                actions_seq, latents_seq = agent.plan_with_predicted_latents(
                    obs_tensor, task=task, horizon=args.plan_horizon, eval_mode=True
                )
                action = actions_seq[0].detach().cpu().numpy()
                next_obs, reward, done, _ = env.step(action)

                z_pred_next = latents_seq[1]
                next_obs_tensor = torch.as_tensor(next_obs, device=agent.device, dtype=torch.float32)
                z_real_next = agent.model.encode(next_obs_tensor.unsqueeze(0), task)
                a_plan_next = actions_seq[1] if actions_seq.shape[0] > 1 else actions_seq[0]
                distance = torch.norm(z_real_next.squeeze(0) - z_pred_next).item()

                feat = torch.cat(
                    [
                        z_real_next.squeeze(0).detach().cpu(),
                        z_pred_next.detach().cpu(),
                        (z_real_next.squeeze(0) - z_pred_next).detach().cpu(),
                        a_plan_next.detach().cpu(),
                    ],
                    dim=-1,
                )
                history.append(feat)

                if ep_steps % args.teacher_interval == 0 and distance >= args.min_distance:
                    a_teacher = agent.plan_from_observation(next_obs_tensor, task=task, eval_mode=True)
                    history_feats = pad_history(history, args.history_len, feat_dim)
                    buffer.add(
                        z_real_next.squeeze(0),
                        z_pred_next,
                        a_plan_next,
                        a_teacher.squeeze(0),
                        distance,
                        history_feats,
                    )
                obs = next_obs
                ep_steps += 1
                total_steps += 1
                if max_samples is not None and len(buffer) >= max_samples:
                    break
            episodes += 1
            mean_dist = (torch.tensor(buffer.distance).mean().item() if len(buffer) > 0 else 0.0)
            print(
                f"[ep {episodes}] steps: {ep_steps}, collected: {len(buffer)} samples, "
                f"mean distance: {mean_dist:.4f}"
            )
    except KeyboardInterrupt:
        print("Interrupted; saving collected samples so far...")

    output_dir = os.path.dirname(output_path) or "."
    os.makedirs(output_dir, exist_ok=True)
    data = buffer.to_tensor_dict(device="cpu")
    torch.save(data, output_path)
    elapsed = time.time() - start_time
    steps_per_sec = total_steps / max(elapsed, 1e-6)
    print(f"Saved {len(buffer)} samples to {output_path} ({steps_per_sec:.1f} env steps/sec)")


def _resolve_models(args: argparse.Namespace) -> Iterable[tuple[str, str]]:
    ckpts = list_pretrained_checkpoints(args.checkpoint_dir, exclude_patterns=args.exclude_pattern)
    if args.all_models or (not args.model_id and not args.checkpoint):
        if not ckpts:
            raise ValueError(f"No checkpoints found in {args.checkpoint_dir}")
        return ckpts.items()
    if args.model_id:
        if args.model_id not in ckpts:
            raise ValueError(
                f"Model id '{args.model_id}' not found in {args.checkpoint_dir}. Available: {list(ckpts.keys())}"
            )
        return [(args.model_id, ckpts[args.model_id])]
    if args.checkpoint:
        model_id = Path(args.checkpoint).stem
        return [(model_id, args.checkpoint)]
    raise ValueError("Provide --model_id, --all_models, or --checkpoint for manual path.")


def _load_agent_for_model(model_id: str, ckpt_path: str, args: argparse.Namespace, device: torch.device):
    spec_overrides = {"spec_enabled": False, "speculate": False}
    agent, cfg, _ = load_pretrained_tdmpc2(
        model_id,
        checkpoint_path=ckpt_path,
        device=str(device),
        task=args.task,
        config_path=args.config,
        spec_overrides=spec_overrides,
    )
    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        agent.model = wrap_dataparallel(agent.model)
    return agent, cfg


def main_worker(rank: int, world_size: int, args: argparse.Namespace) -> None:
    del world_size  # unused; collection runs in a single process

    set_seed(args.seed)
    use_gpu = torch.cuda.is_available() and not args.device.startswith("cpu")
    device = torch.device("cuda" if use_gpu else "cpu")

    for model_id, ckpt_path in _resolve_models(args):
        agent, cfg = _load_agent_for_model(model_id, ckpt_path, args, device)
        out_path = args.output
        if args.all_models or (args.model_id and args.output == DEFAULT_OUTPUT) or not args.model_id:
            base_dir = os.path.dirname(args.output) or os.path.dirname(DEFAULT_OUTPUT) or "data"
            filename = f"corrector_data_{model_id}.pt"
            out_path = os.path.join(base_dir, filename)
        print(f"[collect_corrector_data] model_id={model_id} checkpoint={ckpt_path}")
        collect_for_agent(agent, cfg, args, out_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", "--env", dest="task", type=str, help="Task name / env id", required=False)
    parser.add_argument("--checkpoint", type=str, required=False, default=None, help="Manual TD-MPC2 checkpoint path")
    parser.add_argument("--checkpoint_dir", type=str, default="tdmpc2_pretrained", help="Directory containing pretrained checkpoints")
    parser.add_argument("--model_id", type=str, default=None, help="Model id (checkpoint stem) to load")
    parser.add_argument("--all_models", action="store_true", help="Iterate over all checkpoints in checkpoint_dir")
    parser.add_argument(
        "--exclude_pattern",
        action="append",
        help="Optional substring(s) to skip when discovering checkpoints",
    )
    parser.add_argument("--episodes", type=int, default=50, help="Number of episodes to collect")
    parser.add_argument("--max_steps", type=int, default=None, help="Max steps per episode")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", type=str, default=DEFAULT_OUTPUT, help="Path to save dataset")
    parser.add_argument("--min_distance", type=float, default=0.0, help="Minimum latent distance to record")
    parser.add_argument("--max_samples", type=int, default=-1, help="Maximum number of samples to collect")
    parser.add_argument("--seed", type=int, default=1, help="Random seed")
    parser.add_argument("--config", type=str, default=None, help="Optional path to config.yaml override")
    parser.add_argument("--history_len", type=int, default=4, help="History length for temporal features")
    parser.add_argument("--plan_horizon", type=int, default=3, help="Teacher planning horizon")
    parser.add_argument("--teacher_interval", type=int, default=1, help="Collect teacher action every N steps")
    parser.add_argument(
        "--gpus", type=str, default="1", help="GPU selection: 'all', N, or comma-separated list"
    )
    return parser.parse_args()


if __name__ == "__main__":
    launch(parse_args(), main_worker, use_ddp=False, allow_dataparallel=True)
