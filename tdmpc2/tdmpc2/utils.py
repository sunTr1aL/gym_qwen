"""Utility helpers for loading pretrained TD-MPC2 checkpoints and metadata."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from omegaconf import OmegaConf

from tdmpc2.common.parser import parse_cfg
from tdmpc2.common.seed import set_seed

_CANONICAL_SIZE = {1: "1m", 5: "5m", 19: "19m", 48: "48m", 317: "317m"}
_SIZE_ALIAS = {
    "1": 1,
    "1m": 1,
    "5": 5,
    "5m": 5,
    "10": 10,  # reserve for potential future manifests
    "10m": 10,
    "19": 19,
    "19m": 19,
    "48": 48,
    "48m": 48,
    "300": 317,
    "300m": 317,
    "317": 317,
    "317m": 317,
}

DEFAULT_PRETRAINED_URLS: Dict[str, str] = {
    # Official TD-MPC2 release URLs (excluding the ~1M parameter variant)
    # Users can override these via a manifest or by dropping checkpoints into
    # `model_dir` with the naming scheme `tdmpc2_<size>.pt|.pth`.
    "5m": "https://www.tdmpc2.com/models/tdmpc2_5m.pt",
    "19m": "https://www.tdmpc2.com/models/tdmpc2_19m.pt",
    "48m": "https://www.tdmpc2.com/models/tdmpc2_48m.pt",
    "317m": "https://www.tdmpc2.com/models/tdmpc2_317m.pt",
}


def _normalize_size_name(model_size: str) -> Tuple[str, int]:
    key = str(model_size).lower()
    if key not in _SIZE_ALIAS:
        raise ValueError(
            f"Unknown model size '{model_size}'. Expected one of {_SIZE_ALIAS.keys()}"
        )
    numeric = _SIZE_ALIAS[key]
    canonical = _CANONICAL_SIZE.get(numeric, f"{numeric}m")
    return canonical, numeric


def available_pretrained_sizes(model_dir: str, include_smallest: bool = False) -> List[str]:
    """Return discovered pretrained sizes under ``model_dir``.

    Expects filenames of the form ``tdmpc2_<size>.pt`` or ``tdmpc2_<size>.pth``.
    Falls back to an empty list if no files are present.
    """

    pattern = re.compile(r"tdmpc2_([a-z0-9]+)\.(?:pt|pth)")
    sizes: List[str] = []
    for path in Path(model_dir).glob("tdmpc2_*.*"):
        match = pattern.match(path.name)
        if not match:
            continue
        size_name = match.group(1).lower()
        if size_name not in _SIZE_ALIAS:
            continue
        canonical, numeric = _normalize_size_name(size_name)
        if not include_smallest and numeric == 1:
            continue
        if canonical not in sizes:
            sizes.append(canonical)
    return sorted(sizes, key=lambda s: _SIZE_ALIAS[s])


def resolve_pretrained_checkpoint(model_size: str, model_dir: str) -> Path:
    canonical, _ = _normalize_size_name(model_size)
    candidates = [
        Path(model_dir) / f"tdmpc2_{canonical}.pt",
        Path(model_dir) / f"tdmpc2_{canonical}.pth",
    ]
    for cand in candidates:
        if cand.exists():
            return cand
    raise FileNotFoundError(
        f"No checkpoint found for size '{canonical}' in {model_dir}. "
        "Expected tdmpc2_<size>.pt or .pth."
    )


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
    model_size: str,
    model_dir: str = "tdmpc2_pretrained",
    device: str = "cpu",
    task: Optional[str] = None,
    config_path: Optional[str] = None,
    spec_overrides: Optional[Dict] = None,
    corrector_ckpt: Optional[str] = None,
):
    """Instantiate a TD-MPC2 agent from a pretrained checkpoint.

    Args:
        model_size: Canonical size string (e.g., ``"5m"``) or alias.
        model_dir: Directory containing downloaded checkpoints.
        device: Target device string.
        task: Optional task override. Falls back to checkpoint metadata.
        config_path: Optional path to ``config.yaml`` to load as the base config.
        spec_overrides: Dict of spec-related overrides applied before parsing.
        corrector_ckpt: Optional corrector checkpoint to attach for evaluation.

    Returns:
        (agent, cfg, metadata) tuple with the loaded, eval-mode TD-MPC2 agent.
    """

    canonical, numeric_size = _normalize_size_name(model_size)
    ckpt_path = resolve_pretrained_checkpoint(canonical, model_dir)
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    metadata = state.get("metadata", {}) if isinstance(state, dict) else {}

    cfg_file = (
        Path(config_path)
        if config_path is not None
        else Path(__file__).resolve().parent / "config.yaml"
    )
    cfg = OmegaConf.load(cfg_file)
    cfg.device = device
    cfg.task = _infer_task_from_metadata(metadata, task) or cfg.get("task")
    cfg.model_size = metadata.get("model_size", numeric_size)
    cfg.checkpoint = str(ckpt_path)
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
    # Local import to avoid circular imports when importing module-level
    # helpers (e.g., DEFAULT_PRETRAINED_URLS) from scripts.
    from tdmpc2.tdmpc2 import TDMPC2
    agent = TDMPC2(cfg)
    agent.load(ckpt_path)
    agent.eval()
    return agent, cfg, metadata


__all__ = [
    "DEFAULT_PRETRAINED_URLS",
    "available_pretrained_sizes",
    "load_pretrained_tdmpc2",
    "resolve_pretrained_checkpoint",
]
