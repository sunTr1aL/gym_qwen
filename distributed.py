"""Lightweight DistributedDataParallel helpers (re-exported for backward use)."""

from launch import cleanup_ddp, ddp_available, setup_ddp

__all__ = ["setup_ddp", "cleanup_ddp", "ddp_available"]
