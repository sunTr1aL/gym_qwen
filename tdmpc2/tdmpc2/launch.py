"""Lightweight launch utilities for single-process or simple DataParallel runs.

These helpers provide compatibility shims for legacy TD-MPC2 scripts without
relying on an external launcher module. Distributed training is not enabled;
`launch` simply calls the provided worker locally.
"""
from __future__ import annotations

import os
from typing import Any, Callable

import torch


def ddp_available(*_: Any, **__: Any) -> bool:
    """Return False to indicate DDP is not configured in this lightweight shim."""

    return False


def setup_ddp(*args: Any, **kwargs: Any) -> None:  # pragma: no cover - shim
    """Placeholder for DDP setup (no-op)."""

    del args, kwargs
    return None


def cleanup_ddp() -> None:  # pragma: no cover - shim
    """Placeholder for DDP cleanup (no-op)."""

    return None


def wrap_dataparallel(module: torch.nn.Module) -> torch.nn.Module:
    """Wrap a module in DataParallel when multiple GPUs are visible."""

    if torch.cuda.is_available():
        visible = os.environ.get("CUDA_VISIBLE_DEVICES")
        num_gpus = torch.cuda.device_count()
        if visible:
            # Filter out masked devices
            num_gpus = len(visible.split(","))
        if num_gpus > 1:
            return torch.nn.DataParallel(module)
    return module


def launch(
    args: Any,
    main_fn: Callable[[int, int, Any], Any],
    use_ddp: bool = False,
    allow_dataparallel: bool = True,
    **_: Any,
) -> Any:
    """Call the provided worker locally.

    Parameters mirror the original API for compatibility; only rank 0 / world size 1
    is used here.
    """

    result = main_fn(rank=0, world_size=1, args=args)
    if allow_dataparallel:
        return result
    return result
