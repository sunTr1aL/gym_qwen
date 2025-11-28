"""Checkpoint discovery and loading utilities for pretrained TD-MPC2 agents."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from omegaconf import OmegaConf

from tdmpc2 import TDMPC2
from tdmpc2.common.parser import parse_cfg, populate_env_dims
from tdmpc2.common.seed import set_seed


def list_pretrained_checkpoints(
    checkpoint_dir: str = "tdmpc2_pretrained",
    extensions: List[str] | Tuple[str, ...] = (".pt", ".pth", ".ckpt"),
    model_size_filter: Optional[str] = None,
) -> Dict[str, Dict[str, str]]:
    """Discover pretrained checkpoints and parse model metadata from filenames.

    Expected filename pattern: ``<model_name>-<model_size>.pt`` (stem used as
    ``model_id``). The returned mapping is:

    ``model_id -> {"path": path, "model_name": model_name, "model_size": model_size}``

    ``model_size_filter`` performs a case-insensitive exact match on the
    ``model_size`` field when provided.
    """

    checkpoints: Dict[str, Dict[str, str]] = {}
    root = Path(checkpoint_dir)
    for path in root.glob("*"):
        if not path.is_file() or path.suffix.lower() not in extensions:
            continue
        stem = path.stem
        parts = stem.split("-")
        if len(parts) == 2:
            model_name, model_size = parts[0], parts[1]
        else:
            model_name, model_size = stem, ""
        if model_size_filter is not None and model_size.lower() != model_size_filter.lower():
            continue
        checkpoints[stem] = {
            "path": str(path),
            "model_name": model_name,
            "model_size": model_size,
        }
    return dict(sorted(checkpoints.items()))


def _infer_task_from_metadata(metadata: Dict, fallback: Optional[str]) -> Optional[str]:
    if fallback:
        return fallback
    if metadata.get("task"):
        return metadata["task"]
    tasks = metadata.get("tasks")
    if isinstance(tasks, (list, tuple)) and tasks:
        return tasks[0]
    return None


def _apply_multitask_metadata(cfg, metadata: Dict) -> None:
    tasks = metadata.get("tasks")
    task_dim = metadata.get("task_dim")
    if tasks:
        cfg.force_multitask = True
        cfg.tasks_override = list(tasks)
        if task_dim is not None:
            cfg.task_dim = task_dim


def _is_placeholder(val) -> bool:
    return val is None or (isinstance(val, str) and val.strip() == "???")


def _normalize_pretrained_cfg_numeric_fields(cfg, env_for_dims=None):
    """Normalize numeric fields when loading pretrained checkpoints outside Hydra."""

    if hasattr(cfg, "discount_denom"):
        val = cfg.discount_denom
        default_denom = 5
        if isinstance(val, str):
            if val.strip() == "???":
                cfg.discount_denom = default_denom
            else:
                try:
                    cfg.discount_denom = int(val)
                except Exception:
                    cfg.discount_denom = default_denom
        elif isinstance(val, float):
            cfg.discount_denom = int(val)

    if hasattr(cfg, "episode_lengths"):
        ep = cfg.episode_lengths
        new_eps = None

        if isinstance(ep, (list, tuple)):
            tmp = []
            for v in ep:
                if isinstance(v, str):
                    if v.strip() == "???":
                        continue
                    try:
                        tmp.append(int(v))
                    except Exception:
                        pass
                elif isinstance(v, (int, float)):
                    tmp.append(int(v))
            if tmp:
                new_eps = tmp

        elif isinstance(ep, str):
            s = ep.strip()
            if s == "???":
                new_eps = None
            else:
                parts = [x.strip() for x in s.split(",")]
                tmp = []
                for p in parts:
                    if p.isdigit():
                        tmp.append(int(p))
                if tmp:
                    new_eps = tmp

        elif isinstance(ep, (int, float)):
            new_eps = [int(ep)]

        if new_eps is None:
            if env_for_dims is not None and hasattr(env_for_dims, "_max_episode_steps"):
                default_len = int(env_for_dims._max_episode_steps)
            elif env_for_dims is not None and hasattr(env_for_dims, "episode_length"):
                default_len = int(env_for_dims.episode_length)
            else:
                default_len = 1000

            tasks = getattr(cfg, "tasks", None)
            if isinstance(tasks, (list, tuple)):
                num_tasks = len(tasks)
            else:
                num_tasks = 1
            new_eps = [default_len] * num_tasks

        cfg.episode_lengths = new_eps


def load_pretrained_tdmpc2(
    model_id: str,
    checkpoint_path: str,
    device: str = "cuda",
    task: Optional[str] = None,
    config_path: Optional[str] = None,
    spec_overrides: Optional[Dict] = None,
    corrector_ckpt: Optional[str] = None,
):
    """Instantiate a TD-MPC2 agent from an arbitrary checkpoint path.

    Args:
        model_id: Filename stem for the checkpoint (used for logging/metadata).
        checkpoint_path: Path to the pretrained weights.
        device: Target device string.
        task: Optional task override. Falls back to checkpoint metadata.
        config_path: Optional path to ``config.yaml`` to load as the base config.
        spec_overrides: Dict of spec-related overrides applied before parsing.
        corrector_ckpt: Optional corrector checkpoint to attach for evaluation.

    Returns:
        (agent, cfg, metadata) tuple with the loaded, eval-mode TD-MPC2 agent.
    """

    state = torch.load(checkpoint_path, map_location=device, weights_only=False)
    metadata = state.get("metadata", {}) if isinstance(state, dict) else {}

    ckpt_cfg = None
    if isinstance(state, dict):
        for key in ("cfg", "config", "hydra_cfg"):
            if key in state:
                ckpt_cfg = state[key]
                break

    if ckpt_cfg is not None:
        cfg_container = OmegaConf.to_container(ckpt_cfg, resolve=True)
        cfg = OmegaConf.create(cfg_container)
        cfg.device = device
        cfg.model_id = model_id
        cfg.checkpoint = str(checkpoint_path)
        cfg.use_corrector = bool(corrector_ckpt)
        cfg.corrector_ckpt = corrector_ckpt
        cfg.corrector_type = metadata.get("corrector_type", cfg.get("corrector_type"))
        cfg.speculate = False
        cfg.spec_enabled = False
        cfg.save_video = False
        cfg.compile = False
        cfg.disable_wandb = True
        if spec_overrides:
            for k, v in spec_overrides.items():
                setattr(cfg, k, v)
        cfg = parse_cfg(cfg)

        needs_env = any(
            _is_placeholder(getattr(cfg, field, None)) for field in ["obs_shape", "obs_dim", "action_dim", "action_dims"]
        )
        env_for_dims = None
        if needs_env:
            cfg, env_for_dims = populate_env_dims(cfg)
        _normalize_pretrained_cfg_numeric_fields(cfg, env_for_dims=env_for_dims)
    else:
        cfg_file = (
            Path(config_path)
            if config_path is not None
            else Path(__file__).resolve().parent / "config.yaml"
        )
        cfg = OmegaConf.load(cfg_file)
        cfg.device = device
        cfg.task = _infer_task_from_metadata(metadata, task) or cfg.get("task")
        cfg.model_id = model_id
        cfg.checkpoint = str(checkpoint_path)
        cfg.use_corrector = bool(corrector_ckpt)
        cfg.corrector_ckpt = corrector_ckpt
        cfg.corrector_type = metadata.get("corrector_type", cfg.get("corrector_type"))
        cfg.speculate = False
        cfg.spec_enabled = False
        cfg.save_video = False
        cfg.compile = False
        cfg.disable_wandb = True
        _apply_multitask_metadata(cfg, metadata)
        if spec_overrides:
            for k, v in spec_overrides.items():
                setattr(cfg, k, v)
        cfg = parse_cfg(cfg)
        cfg, env_for_dims = populate_env_dims(cfg)
        _normalize_pretrained_cfg_numeric_fields(cfg, env_for_dims=env_for_dims)

    model_state = None
    if isinstance(state, dict):
        if "model" in state:
            model_state = state["model"]
        elif "agent" in state and isinstance(state["agent"], dict):
            model_state = state["agent"].get("model", state["agent"])
        elif "state_dict" in state and isinstance(state["state_dict"], dict):
            model_state = state["state_dict"]

    if model_state is None:
        model_state = state

    obs_shape = getattr(cfg, "obs_shape", None)
    if _is_placeholder(obs_shape):
        raise ValueError("Pretrained config is missing obs_shape after normalization")
    action_dim_val = getattr(cfg, "action_dim", None)
    if _is_placeholder(action_dim_val):
        raise ValueError("Pretrained config is missing action_dim after normalization")

    if cfg.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available; choose cpu device instead.")

    set_seed(cfg.seed)
    agent = TDMPC2(cfg)

    expected_state = agent.model.state_dict()
    if isinstance(model_state, dict):
        sample_key = next(iter(expected_state.keys()))
        if sample_key in model_state and hasattr(model_state[sample_key], "shape"):
            if expected_state[sample_key].shape != model_state[sample_key].shape:
                raise ValueError(
                    f"Shape mismatch for parameter '{sample_key}': expected {expected_state[sample_key].shape}, "
                    f"found {model_state[sample_key].shape} in checkpoint"
                )

    agent.load(model_state)
    agent.eval()
    return agent, cfg, metadata


__all__ = ["list_pretrained_checkpoints", "load_pretrained_tdmpc2"]
