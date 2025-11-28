"""Checkpoint discovery and loading utilities for pretrained TD-MPC2 agents."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from omegaconf import OmegaConf

from tdmpc2 import TDMPC2


def _infer_arch_dims_from_state_dict(state: dict):
    """Infer architecture-relevant dimensions directly from a state dict.

    This uses checkpoint weights as the source of truth for multi-task models so
    we can align the constructed architecture with the saved parameters even
    when YAML defaults drift (e.g., task embedding width).
    """

    task_dim = None
    num_tasks = None
    action_dim = None

    if not isinstance(state, dict):
        return task_dim, num_tasks, action_dim

    task_emb = state.get("_task_emb.weight")
    if isinstance(task_emb, torch.Tensor):
        num_tasks, task_dim = task_emb.shape

    action_masks = state.get("_action_masks")
    if isinstance(action_masks, torch.Tensor):
        num_tasks = action_masks.shape[0]
        action_dim = action_masks.shape[1]

    return task_dim, num_tasks, action_dim


def _apply_state_arch_overrides(cfg, state: dict, model_id: Optional[str] = None):
    """Align architecture-critical cfg fields with checkpoint state shapes.

    The official multi-task checkpoints were trained with specific task/action
    dimensions (e.g., 64-d task embeddings for mt30). Later config defaults can
    drift (96-d embeddings), so we infer the saved dimensions from the state
    dict and override the cfg before instantiating the model to avoid shape
    mismatches during load_state_dict.
    """

    inferred_task_dim, inferred_num_tasks, inferred_action_dim = _infer_arch_dims_from_state_dict(state)

    if inferred_task_dim is not None:
        cfg.task_dim = int(inferred_task_dim)

    if inferred_action_dim is not None:
        cfg.action_dim = int(inferred_action_dim)

    if getattr(cfg, "multitask", False):
        tasks = getattr(cfg, "tasks", None)
        if isinstance(tasks, (list, tuple)):
            task_count = len(tasks)
        else:
            task_count = inferred_num_tasks or 1
        cfg.action_dims = [int(cfg.action_dim)] * task_count

    q_in_dim = cfg.latent_dim + cfg.action_dim + cfg.task_dim
    debug_model_id = model_id or getattr(cfg, "model_id", None) or "<unknown>"
    print(
        f"[load_pretrained_tdmpc2] model_id={debug_model_id} "
        f"latent_dim={cfg.latent_dim} action_dim={cfg.action_dim} "
        f"task_dim={cfg.task_dim} -> Q-input={q_in_dim} "
        f"(checkpoint task_dim={inferred_task_dim}, action_dim={inferred_action_dim})"
    )



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

    model_state = _extract_state_dict_from_checkpoint(state)

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

            _apply_state_arch_overrides(cfg, model_state, model_id=model_id)

            agent = TDMPC2(cfg)
            agent.to(device)
            agent.eval()

            agent.load(model_state)
            return agent, cfg

    # Fallback: no config stored in the checkpoint; load from YAML based on model id.
    if model_id is None:
        model_id = Path(checkpoint_path).stem
    model_id = _canonical_model_id(model_id)

    from tdmpc2.common.parser import parse_cfg, populate_env_dims

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
        cfg, _ = populate_env_dims(cfg)

    _apply_state_arch_overrides(cfg, model_state, model_id=model_id)

    agent = TDMPC2(cfg)
    agent.to(device)
    agent.eval()

    agent.load(model_state)

    return agent, cfg


__all__ = ["list_pretrained_checkpoints", "load_pretrained_tdmpc2"]
