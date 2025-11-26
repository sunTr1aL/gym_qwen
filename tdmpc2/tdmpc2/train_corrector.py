import argparse
import time
from pathlib import Path
from typing import Dict, List

import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

from tdmpc2.corrector import build_corrector_from_cfg, corrector_loss
from tdmpc2.launch import cleanup_ddp, ddp_available, launch, setup_ddp, wrap_dataparallel


class CorrectorDataset(Dataset):
    def __init__(self, data: Dict[str, torch.Tensor], filter_min_distance: float = 0.0):
        super().__init__()
        mask = data["distance"] >= filter_min_distance
        self.z_real = data["z_real"][mask]
        self.z_pred = data["z_pred"][mask]
        self.a_plan = data["a_plan"][mask]
        self.a_teacher = data["a_teacher"][mask]

        hist = data.get("history_feats")
        if hist is not None:
            self.history = hist[mask].to(self.z_real.device)
        else:
            self.history = None
        self.history_shape = self.history.shape[1:] if self.history is not None else None
        # Placeholder used when no history is available to avoid None collation.
        self.empty_history = torch.empty((0, 0), device=self.z_real.device, dtype=self.z_real.dtype)

    def __len__(self) -> int:
        return self.z_real.shape[0]

    def __getitem__(self, idx):
        hist = self.history[idx] if self.history is not None else self.empty_history
        return self.z_real[idx], self.z_pred[idx], self.a_plan[idx], self.a_teacher[idx], hist


_ACTION_FALLBACK_WARNED = False


def _select_action_tensor(state: Dict[str, torch.Tensor]) -> torch.Tensor:
    """Return the planned/speculative action tensor with legacy compatibility."""

    global _ACTION_FALLBACK_WARNED
    if "a_plan" in state:
        return state["a_plan"]
    if "a_spec" in state:
        if not _ACTION_FALLBACK_WARNED:
            print("WARNING: 'a_plan' missing in buffer, using 'a_spec' instead.")
            _ACTION_FALLBACK_WARNED = True
        return state["a_spec"]
    raise ValueError("Missing action key in buffer: expected 'a_plan' or 'a_spec'.")


def _to_tensor(value, device: torch.device) -> torch.Tensor:
    """Convert mixed list/scalar entries from saved buffers into tensors."""

    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, list):
        if len(value) == 0:
            return torch.empty(0, device=device)
        first = value[0]
        if isinstance(first, torch.Tensor):
            return torch.stack(value).to(device)
        return torch.as_tensor(value, device=device)
    return torch.as_tensor(value, device=device)


def load_tensor_dict(path: Path, device: torch.device) -> Dict[str, torch.Tensor]:
    state = torch.load(path, map_location=device)
    if isinstance(state, list):
        raise ValueError("Expected a dict of tensors in the dataset file.")

    required = {"z_real", "z_pred", "a_teacher"}
    missing = required - set(state.keys())
    if missing:
        raise ValueError(f"Missing keys in buffer: {missing}")

    action_tensor = _select_action_tensor(state)

    processed = {k: _to_tensor(v, device) for k, v in state.items()}
    processed["a_plan"] = _to_tensor(action_tensor, device)
    # Preserve optional a_spec for compatibility across mixed datasets.
    if "a_spec" not in processed:
        processed["a_spec"] = processed["a_plan"]
    return processed


def maybe_concat(paths: List[Path], device: torch.device) -> Dict[str, torch.Tensor]:
    tensors = [load_tensor_dict(p, device) for p in paths]
    if len(tensors) == 1:
        return tensors[0]
    keys = tensors[0].keys()
    return {k: torch.cat([t[k] for t in tensors], dim=0) for k in keys}


def train_worker(rank: int, world_size: int, args: argparse.Namespace) -> None:
    use_gpu = torch.cuda.is_available() and not args.device.startswith("cpu")
    use_ddp = world_size > 1 and ddp_available(world_size) and use_gpu
    device = (
        setup_ddp(rank, world_size)
        if use_ddp
        else torch.device("cuda" if use_gpu else "cpu")
    )

    data_path = Path(args.data)
    if data_path.is_dir():
        tensors = maybe_concat(sorted(data_path.glob("*.pt")), device)
    else:
        tensors = load_tensor_dict(data_path, device)

    latent_dim = tensors["z_real"].shape[-1]
    act_dim = tensors["a_plan"].shape[-1]

    class Cfg:
        pass

    cfg = Cfg()
    cfg.corrector_type = args.corrector_type
    cfg.corrector_hidden_dim = args.hidden_dim
    cfg.corrector_layers = args.num_layers
    cfg.corrector_tanh_output = True
    cfg.spec_history_len = args.history_len

    dataset = CorrectorDataset(tensors, filter_min_distance=args.filter_min_distance)
    sampler = (
        DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
        if use_ddp
        else None
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=sampler is None,
        sampler=sampler,
        num_workers=4,
        pin_memory=torch.cuda.is_available(),
    )

    corrector = build_corrector_from_cfg(cfg, latent_dim=latent_dim, act_dim=act_dim, device=device)
    if use_ddp:
        corrector = torch.nn.parallel.DistributedDataParallel(
            corrector, device_ids=[rank], output_device=rank, find_unused_parameters=False
        )
    elif torch.cuda.is_available():
        # DataParallel fallback when multiple visible GPUs but DDP disabled.
        dp_ids = list(range(torch.cuda.device_count()))
        if len(dp_ids) > 1:
            corrector = wrap_dataparallel(corrector, device_ids=dp_ids)

    optim = torch.optim.Adam(corrector.parameters(), lr=args.lr)

    global_start = time.time()
    for epoch in range(args.epochs):
        if sampler is not None:
            sampler.set_epoch(epoch)
        total_loss = 0.0
        count = 0
        epoch_start = time.time()
        for batch in loader:
            z_real, z_pred, a_plan, a_teacher, history = batch
            z_real = z_real.to(device, non_blocking=True)
            z_pred = z_pred.to(device, non_blocking=True)
            a_plan = a_plan.to(device, non_blocking=True)
            a_teacher = a_teacher.to(device, non_blocking=True)
            if args.corrector_type == "temporal":
                if history.numel() == 0:
                    raise ValueError(
                        "Temporal corrector requires 'history_feats' in the dataset; received empty history."
                    )
                mismatch_history = history.to(device, non_blocking=True)
            else:
                mismatch_history = None
            optim.zero_grad()
            a_corr = corrector(z_real, z_pred, a_plan, mismatch_history=mismatch_history)
            loss = corrector_loss(a_corr, a_teacher, a_plan, reg_lambda=args.reg_lambda)
            loss.backward()
            optim.step()
            total_loss += loss.item() * z_real.shape[0]
            count += z_real.shape[0]
        if rank == 0:
            avg_loss = total_loss / max(count, 1)
            elapsed = time.time() - epoch_start
            samples_per_sec = count / max(elapsed, 1e-6)
            print(
                f"Epoch {epoch+1}/{args.epochs} - loss: {avg_loss:.4f} - "
                f"samples/sec: {samples_per_sec:.1f}"
            )

    if rank == 0:
        state = corrector.module.state_dict() if hasattr(corrector, "module") else corrector.state_dict()
        torch.save({"corrector": state, "corrector_type": args.corrector_type}, args.save_path)
        total_time = time.time() - global_start
        print(
            f"Saved trained corrector to {args.save_path} after {args.epochs} epochs "
            f"({len(dataset) * args.epochs / max(total_time, 1e-6):.1f} samples/sec)."
        )

    if use_ddp:
        cleanup_ddp()


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a speculative action corrector offline.")
    parser.add_argument("--data", type=str, required=True, help="Path to saved corrector buffer (.pt) or directory of buffers")
    parser.add_argument("--tdmpc_ckpt", type=str, required=False, help="TD-MPC2 checkpoint to derive dimensions", default=None)
    parser.add_argument("--save_path", type=str, default="corrector.pth", help="Output path for trained weights")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--reg_lambda", type=float, default=0.0, help="Residual L2 regularization weight")
    parser.add_argument("--filter_min_distance", type=float, default=0.0, help="Minimum distance to keep a sample")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--corrector_type", type=str, default="two_tower", choices=["two_tower", "temporal"])
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--history_len", type=int, default=4)
    parser.add_argument(
        "--gpus", type=str, default="1", help="GPU selection: 'all', N, or comma-separated list"
    )
    args = parser.parse_args()

    launch(args, train_worker, use_ddp=True, allow_dataparallel=True)




if __name__ == "__main__":
    main()
