"""Checkpoint discovery and loading utilities for pretrained TD-MPC2 agents."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from omegaconf import OmegaConf

from tdmpc2 import TDMPC2


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
    """Instantiate a TD-MPC2 agent from a checkpoint using its saved config."""

    state = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if not isinstance(state, dict):
        raise RuntimeError(f"Checkpoint {checkpoint_path} must be a mapping with a saved config.")

    metadata = state.get("metadata", {})
    cfg = None
    for key in ("cfg", "config", "hydra_cfg"):
        if key in state:
            cfg = state[key]
            break
    if cfg is None:
        raise RuntimeError(f"Checkpoint {checkpoint_path} does not contain a 'cfg' entry.")

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

    if "model_state" in state:
        model_state = state["model_state"]
    elif "model" in state:
        model_state = state["model"]
    elif "agent" in state and isinstance(state["agent"], dict):
        model_state = state["agent"].get("model", state["agent"])
    elif "state_dict" in state and isinstance(state["state_dict"], dict):
        model_state = state["state_dict"]
    else:
        raise RuntimeError(f"Checkpoint {checkpoint_path} has no recognizable model state.")

    expected_state = agent.model.state_dict()
    sample_key = None
    if isinstance(model_state, dict):
        for key in expected_state:
            if key in model_state:
                sample_key = key
                break
    if sample_key is not None:
        print("New model weight shape:", expected_state[sample_key].shape)
        print("Checkpoint weight shape:", model_state[sample_key].shape)

    agent.load(model_state)
    return agent, cfg


__all__ = ["list_pretrained_checkpoints", "load_pretrained_tdmpc2"]
