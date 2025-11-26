"""Lightweight DistributedDataParallel helpers (re-exported for backward use)."""

from tdmpc2.launch import cleanup_ddp, ddp_available, setup_ddp

__all__ = ["setup_ddp", "cleanup_ddp", "ddp_available"]
