"""Checkpoint discovery and loading utilities for pretrained TD-MPC2 agents."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from omegaconf import OmegaConf

from tdmpc2 import TDMPC2
from common.parser import parse_cfg, populate_env_dims
from envs import make_env


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


def _infer_dims_from_state(model_state: Dict[str, torch.Tensor]) -> Dict[str, int]:
    """Infer key architectural input dimensions directly from a checkpoint state."""

    dims: Dict[str, int] = {}
    if not isinstance(model_state, dict):
        return dims

    if "_task_emb.weight" in model_state:
        dims["task_emb_dim"] = int(model_state["_task_emb.weight"].shape[1])
    if "_encoder.state.0.weight" in model_state:
        dims["encoder_in_dim"] = int(model_state["_encoder.state.0.weight"].shape[1])
    if "_dynamics.0.weight" in model_state:
        dims["dyn_in_dim"] = int(model_state["_dynamics.0.weight"].shape[1])
    if "_reward.0.weight" in model_state:
        dims["rew_in_dim"] = int(model_state["_reward.0.weight"].shape[1])
    if "_pi.0.weight" in model_state:
        dims["pi_in_dim"] = int(model_state["_pi.0.weight"].shape[1])

    if dims:
        print("[DEBUG] Inferred checkpoint dims:", dims)
    return dims


def align_cfg_with_checkpoint(cfg, model_state: Dict[str, torch.Tensor]):
    """Align architecture-critical config fields with a pretrained checkpoint."""

    inferred_dims = _infer_dims_from_state(model_state)

    task_emb_dim = inferred_dims.get("task_emb_dim", None)
    if task_emb_dim is not None:
        cfg.task_dim = int(task_emb_dim)
        cfg.task_emb_dim = int(task_emb_dim)

    encoder_in_dim = inferred_dims.get("encoder_in_dim", None)
    if encoder_in_dim is not None:
        cfg.encoder_in_dim = int(encoder_in_dim)
        base_obs_dim = encoder_in_dim - int(getattr(cfg, "task_dim", 0))
        if base_obs_dim > 0:
            cfg.obs_dim = int(base_obs_dim)
            obs_type = getattr(cfg, "obs_type", "states")
            cfg.obs_shape = {obs_type: (cfg.obs_dim,)}

    dyn_in_dim = inferred_dims.get("dyn_in_dim", None)
    if dyn_in_dim is not None:
        cfg.dyn_in_dim = int(dyn_in_dim)
        latent_dim = int(getattr(cfg, "latent_dim", 0))
        task_dim = int(getattr(cfg, "task_dim", 0))
        candidate_action_dim = dyn_in_dim - latent_dim - task_dim
        if candidate_action_dim > 0:
            cfg.action_dim = int(candidate_action_dim)
            cfg.action_dims = [cfg.action_dim] * max(1, len(getattr(cfg, "tasks", []) or [None]))

    rew_in_dim = inferred_dims.get("rew_in_dim", None)
    if rew_in_dim is not None:
        cfg.rew_in_dim = int(rew_in_dim)

    pi_in_dim = inferred_dims.get("pi_in_dim", None)
    if pi_in_dim is not None:
        cfg.pi_in_dim = int(pi_in_dim)

    cfg.pretrained_aligned = True
    return cfg


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

    model_state = _extract_state_dict_from_checkpoint(state)

    env_for_dims = None
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

            cfg.compile = False

            cfg = parse_cfg(cfg)
            cfg = align_cfg_with_checkpoint(cfg, model_state)
            cfg, env_for_dims = populate_env_dims(cfg)

            if cfg.multitask:
                make_env(cfg)

            agent = TDMPC2(cfg)
            agent.to(device)
            agent.eval()

            agent.load(model_state)
            return agent, cfg

    # Fallback: no config stored in the checkpoint; load from YAML based on model id.
    if model_id is None:
        model_id = Path(checkpoint_path).stem
    model_id = _canonical_model_id(model_id)

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

    cfg_path = model_config_map.get(model_id, base_config)
    if not cfg_path.is_file():
        raise RuntimeError(f"Config file {cfg_path} for model_id={model_id} not found")

    cfg = OmegaConf.load(str(cfg_path))

    # Infer task and size from the model id.
    stem = Path(model_id).stem
    metadata = state.get("metadata", {}) if isinstance(state, dict) else {}
    cfg.task = metadata.get("task", cfg.get("task", None))
    parts = stem.split("-")
    if cfg.task is None and parts:
        # Treat all but a trailing size token as the task name when no metadata is available.
        potential_size = parts[-1].rstrip("mM")
        if len(parts) > 1 and potential_size.isdigit():
            cfg.task = "-".join(parts[:-1])
        else:
            cfg.task = parts[0]

    if len(parts) > 1:
        size_token = parts[-1].rstrip("mM")
        if size_token.isdigit():
            cfg.model_size = int(size_token)

    cfg.checkpoint = str(checkpoint_path)
    cfg.device = str(device)
    cfg.disable_wandb = True
    cfg.eval_mode = True
    cfg.model_id = model_id

    cfg.compile = False

    cfg = parse_cfg(cfg)
    cfg = align_cfg_with_checkpoint(cfg, model_state)
    cfg, env_for_dims = populate_env_dims(cfg)

    if cfg.multitask:
        make_env(cfg)

    agent = TDMPC2(cfg)
    agent.to(device)
    agent.eval()

    agent.load(model_state)

    return agent, cfg


__all__ = ["list_pretrained_checkpoints", "load_pretrained_tdmpc2"]
