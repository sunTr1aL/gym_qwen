"""Checkpoint discovery and loading utilities for pretrained TD-MPC2 agents."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from omegaconf import OmegaConf

from tdmpc2 import TDMPC2
from tdmpc2.common.parser import parse_cfg
from tdmpc2.common.seed import set_seed


def list_pretrained_checkpoints(
    checkpoint_dir: str = "tdmpc2_pretrained",
    extensions: List[str] | Tuple[str, ...] = (".pt", ".pth", ".ckpt"),
    exclude_patterns: Optional[List[str]] = None,
) -> Dict[str, str]:
    """Return a mapping from ``model_id`` to absolute checkpoint path.

    ``model_id`` is defined as the filename stem (basename without extension).
    Files are discovered by extension; optional ``exclude_patterns`` can drop
    any filename containing a given substring.
    """

    checkpoint_paths: Dict[str, str] = {}
    root = Path(checkpoint_dir)
    for path in root.glob("*"):
        if not path.is_file() or path.suffix.lower() not in extensions:
            continue
        name = path.name
        if exclude_patterns and any(pat in name for pat in exclude_patterns):
            continue
        checkpoint_paths[path.stem] = str(path)
    return dict(sorted(checkpoint_paths.items()))


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
    _apply_multitask_metadata(cfg, metadata)
    if spec_overrides:
        for k, v in spec_overrides.items():
            setattr(cfg, k, v)
    cfg = parse_cfg(cfg)
    if cfg.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available; choose cpu device instead.")

    set_seed(cfg.seed)
    agent = TDMPC2(cfg)
    agent.load(checkpoint_path)
    agent.eval()
    return agent, cfg, metadata


__all__ = ["list_pretrained_checkpoints", "load_pretrained_tdmpc2"]
