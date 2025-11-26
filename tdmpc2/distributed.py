"""Lightweight DistributedDataParallel helpers."""

import torch


def setup_ddp(rank: int, world_size: int) -> None:
    torch.distributed.init_process_group(
        backend="nccl",
        init_method="env://",
        world_size=world_size,
        rank=rank,
    )
    torch.cuda.set_device(rank)


def cleanup_ddp() -> None:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


def ddp_available(world_size: int) -> bool:
    return world_size > 1 and torch.cuda.is_available()


__all__ = ["setup_ddp", "cleanup_ddp", "ddp_available"]
