"""Checkpoint discovery and loading utilities for pretrained TD-MPC2 agents."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from omegaconf import OmegaConf

from tdmpc2 import TDMPC2


def _canonical_model_id(model_id: str) -> str:
    stem = Path(model_id).stem
    parts = stem.split("-")
    if len(parts) > 1:
        size_raw = parts[1]
        size_stripped = size_raw.rstrip("mM")
        if size_stripped.isdigit():
            size_raw = f"{int(size_stripped)}M"
        return f"{parts[0]}-{size_raw}"
    return stem


def _extract_state_dict_from_checkpoint(state: dict):
    """Return the model state dict stored in a checkpoint mapping."""

    if isinstance(state, dict):
        for key in ("model_state", "model", "agent", "state_dict"):
            candidate = state.get(key)
            if candidate is None:
                continue
            if key == "agent" and isinstance(candidate, dict):
                # Some checkpoints nest model weights under an "agent" key.
                nested_model = candidate.get("model")
                if isinstance(nested_model, dict):
                    return nested_model
            if isinstance(candidate, dict):
                return candidate
    # Fallback: assume the entire object is already a state dict.
    return state


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


def load_pretrained_tdmpc2(
    checkpoint_path: str,
    device: str = "cuda",
    model_id: Optional[str] = None,
    **_: Dict,
): 
    """Instantiate a TD-MPC2 agent from a checkpoint using an embedded or YAML config."""

    state = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if not isinstance(state, dict):
        raise RuntimeError(f"Checkpoint {checkpoint_path} must be a mapping with a saved config.")

    if model_id is not None:
        model_id = _canonical_model_id(model_id)

    for key in ("cfg", "config", "hydra_cfg"):
        if key in state:
            cfg = state[key]
            if isinstance(cfg, OmegaConf.DictConfig):
                cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
            cfg.device = str(device)
            cfg.disable_wandb = True
            cfg.eval_mode = True
            cfg.checkpoint = str(checkpoint_path)
            if model_id is not None:
                cfg.model_id = model_id

            agent = TDMPC2(cfg)
            agent.to(device)
            agent.eval()

            model_state = _extract_state_dict_from_checkpoint(state)
            agent.load(model_state)
            return agent, cfg

    # Fallback: no config stored in the checkpoint; load from YAML based on model id.
    if model_id is None:
        model_id = Path(checkpoint_path).stem
    model_id = _canonical_model_id(model_id)

    from tdmpc2.common.parser import parse_cfg
    from tdmpc2.envs import make_env

    checkpoint_dir = Path(__file__).resolve().parent
    base_config = checkpoint_dir / "config.yaml"

    model_config_map = {
        "mt30-5M": base_config,
        "mt30-19M": base_config,
        "mt30-48M": base_config,
        "mt30-317M": base_config,
        "mt70-5M": base_config,
        "mt70-19M": base_config,
        "mt70-48M": base_config,
        "mt70-317M": base_config,
        "mt80-5M": base_config,
        "mt80-19M": base_config,
        "mt80-48M": base_config,
        "mt80-317M": base_config,
    }

    if model_id not in model_config_map:
        raise RuntimeError(f"No YAML config mapped for model_id={model_id}")

    cfg_path = model_config_map[model_id]
    if not cfg_path.is_file():
        raise RuntimeError(f"Config file {cfg_path} for model_id={model_id} not found")

    cfg = OmegaConf.load(str(cfg_path))

    # Infer task and size from the model id.
    stem = Path(model_id).stem
    parts = stem.split("-")
    if parts:
        cfg.task = parts[0]
    if len(parts) > 1:
        size_token = parts[1].rstrip("mM")
        if size_token.isdigit():
            cfg.model_size = int(size_token)

    cfg.checkpoint = str(checkpoint_path)
    cfg.device = str(device)
    cfg.disable_wandb = True
    cfg.eval_mode = True
    cfg.model_id = model_id

    cfg = parse_cfg(cfg)

    # Populate observation and action dimensions before building the agent when missing.
    needs_env_dims = False
    for key in ("obs_shape", "obs_shapes", "action_dim", "action_dims", "episode_length", "episode_lengths"):
        val = getattr(cfg, key, None)
        if val is None or (isinstance(val, str) and val == "???"):
            needs_env_dims = True
            break

    if needs_env_dims:
        env = make_env(cfg)
        try:
            close_fn = getattr(env, "close", None)
            if callable(close_fn):
                close_fn()
        except Exception:
            pass

    agent = TDMPC2(cfg)
    agent.to(device)
    agent.eval()

    model_state = _extract_state_dict_from_checkpoint(state)
    agent.load(model_state)

    return agent, cfg


__all__ = ["list_pretrained_checkpoints", "load_pretrained_tdmpc2"]
