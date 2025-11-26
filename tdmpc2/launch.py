"""Shared multi-GPU launcher utilities.

This module centralizes GPU selection, CUDA visibility, and DDP/DataParallel
launch semantics so training and evaluation scripts do not duplicate logic.
"""

from __future__ import annotations

import os
from typing import Callable, List, Optional, Sequence

import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def parse_gpus_arg(gpus_arg: Optional[object]) -> List[int]:
    """Parse user GPU selection into a list of device ids.

    Supports values like "all", an integer (N → [0, N-1]), or a comma-separated
    list "0,2,3". If CUDA is unavailable, returns an empty list.
    """

    if not torch.cuda.is_available():
        return []

    if gpus_arg is None:
        return [0] if torch.cuda.device_count() > 0 else []

    gpus_str = str(gpus_arg).strip()
    if gpus_str.lower() == "all":
        return list(range(torch.cuda.device_count()))

    if gpus_str.isdigit():
        requested = int(gpus_str)
        if requested <= 0:
            return []
        available = torch.cuda.device_count()
        if requested > available:
            raise ValueError(f"Requested {requested} GPUs but only {available} available")
        return list(range(requested))

    device_ids: List[int] = []
    for token in gpus_str.split(","):
        if not token.strip():
            continue
        if not token.strip().isdigit():
            raise ValueError(f"Invalid GPU token '{token}' in --gpus {gpus_str}")
        device_ids.append(int(token))

    if not device_ids:
        return []

    available = torch.cuda.device_count()
    if max(device_ids) >= available:
        raise ValueError(f"GPU ids {device_ids} exceed available {available}")
    return device_ids


def set_visible_devices(device_ids: Sequence[int]) -> None:
    """Set CUDA_VISIBLE_DEVICES according to the requested ids."""

    if device_ids:
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(i) for i in device_ids)
    else:
        os.environ.pop("CUDA_VISIBLE_DEVICES", None)


def setup_ddp(rank: int, world_size: int) -> torch.device:
    """Initialize DDP for the current process and return its device."""

    if "MASTER_ADDR" not in os.environ:
        # Single-node mp.spawn DDP: pick a stable port for all ranks.
        port = 29500 + (os.getpid() % 2000)
        init_method = f"tcp://127.0.0.1:{port}"
    else:
        init_method = "env://"

    dist.init_process_group(
        backend="nccl", init_method=init_method, world_size=world_size, rank=rank
    )
    torch.cuda.set_device(rank)
    return torch.device(f"cuda:{rank}")


def cleanup_ddp() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def ddp_available(world_size: int) -> bool:
    return world_size > 1 and torch.cuda.is_available() and dist.is_available()


def wrap_dataparallel(model: torch.nn.Module, device_ids: Optional[Sequence[int]] = None):
    if not torch.cuda.is_available():
        return model
    ids = list(device_ids) if device_ids is not None else list(range(torch.cuda.device_count()))
    if len(ids) <= 1:
        return model
    return torch.nn.DataParallel(model, device_ids=ids)


def launch(
    args: object,
    main_fn: Callable[[int, int, object], None],
    use_ddp: bool = True,
    allow_dataparallel: bool = True,
) -> None:
    """Entry-point wrapper that dispatches between single, DDP, or DataParallel."""

    requested_gpus = getattr(args, "gpus", None)
    device_ids = parse_gpus_arg(requested_gpus)
    set_visible_devices(device_ids)
    args.device_ids = device_ids

    visible_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
    print(
        f"[GPU CONFIG] Requested: {requested_gpus}; visible devices after filter: {visible_count};"
        f" selected ids: {device_ids}"
    )

    env_world_size = int(os.environ.get("WORLD_SIZE", "0"))

    # Case A: torchrun provided WORLD_SIZE/LOCAL_RANK
    if env_world_size > 1:
        rank = int(os.environ["LOCAL_RANK"])
        world_size = env_world_size
        if use_ddp and ddp_available(world_size):
            print(f"[DDP/torchrun] rank={rank} world_size={world_size}")
            main_fn(rank, world_size, args)
        else:
            if rank == 0:
                print("[torchrun/no-ddp] Falling back to single-process execution on rank 0")
                main_fn(0, 1, args)
        return

    user_world_size = len(device_ids) if device_ids else 1

    # Case B: manual multi-GPU via python ... --gpus N
    if use_ddp and ddp_available(user_world_size) and user_world_size > 1:
        print(f"[DDP/spawn] Spawning {user_world_size} workers.")
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", "29500")
        mp.spawn(main_fn, args=(user_world_size, args), nprocs=user_world_size, join=True)
        return

    # Case C: DataParallel fallback (single process)
    if allow_dataparallel and user_world_size > 1:
        print(f"[DataParallel] Using GPUs: {device_ids}")
        main_fn(rank=0, world_size=1, args=args)
        return

    # Case D: single GPU/CPU
    print("[Single GPU/CPU] world_size=1")
    main_fn(rank=0, world_size=1, args=args)


__all__ = [
    "parse_gpus_arg",
    "set_visible_devices",
    "setup_ddp",
    "cleanup_ddp",
    "ddp_available",
    "wrap_dataparallel",
    "launch",
]
