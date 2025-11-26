import argparse
import os
import time
from pathlib import Path
from typing import Dict, List, Sequence


def _configure_gpus_from_argv() -> str:
    """Parse --gpus early and set CUDA_VISIBLE_DEVICES before importing torch."""

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--gpus", type=str, default="all")
    args, _ = parser.parse_known_args()
    gpus_arg = args.gpus
    if gpus_arg and gpus_arg != "all":
        visible = gpus_arg if not gpus_arg.isdigit() else ",".join(str(i) for i in range(int(gpus_arg)))
        os.environ["CUDA_VISIBLE_DEVICES"] = visible
    return gpus_arg


_EARLY_GPUS = _configure_gpus_from_argv()

import torch  # noqa: E402
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

from tdmpc2.corrector import build_corrector_from_cfg, corrector_loss
from tdmpc2.distributed import cleanup_ddp, ddp_available, setup_ddp


class CorrectorDataset(Dataset):
    def __init__(self, data: Dict[str, torch.Tensor], filter_min_distance: float = 0.0):
        super().__init__()
        mask = data["distance"] >= filter_min_distance
        self.z_real = data["z_real"][mask]
        self.z_pred = data["z_pred"][mask]
        self.a_plan = data["a_plan"][mask]
        self.a_teacher = data["a_teacher"][mask]
        self.history = data.get("history_feats")

    def __len__(self) -> int:
        return self.z_real.shape[0]

    def __getitem__(self, idx):
        hist = self.history[idx] if self.history is not None else None
        return self.z_real[idx], self.z_pred[idx], self.a_plan[idx], self.a_teacher[idx], hist


def load_tensor_dict(path: Path, device: torch.device) -> Dict[str, torch.Tensor]:
    state = torch.load(path, map_location=device)
    if isinstance(state, list):
        raise ValueError("Expected a dict of tensors in the dataset file.")
    required = {"z_real", "z_pred", "a_plan", "a_teacher"}
    missing = required - set(state.keys())
    if missing:
        raise ValueError(f"Missing keys in buffer: {missing}")
    return {k: (torch.stack(v) if isinstance(v, list) else v).to(device) for k, v in state.items()}


def maybe_concat(paths: List[Path], device: torch.device) -> Dict[str, torch.Tensor]:
    tensors = [load_tensor_dict(p, device) for p in paths]
    if len(tensors) == 1:
        return tensors[0]
    keys = tensors[0].keys()
    return {k: torch.cat([t[k] for t in tensors], dim=0) for k in keys}


def _resolve_device_ids(gpus: str) -> Sequence[int]:
    if not torch.cuda.is_available():
        return []
    if gpus in (None, "all"):
        return list(range(torch.cuda.device_count()))
    if gpus.isdigit():
        requested = int(gpus)
        if requested <= 0:
            return []
        if requested > torch.cuda.device_count():
            raise ValueError(f"Requested {requested} GPUs but only {torch.cuda.device_count()} available")
        return list(range(requested))
    cleaned = [int(g.strip()) for g in gpus.split(",") if g.strip()]
    if not cleaned:
        return []
    if max(cleaned) >= torch.cuda.device_count():
        raise ValueError(f"GPU ids {cleaned} exceed available {torch.cuda.device_count()}")
    return cleaned


def train_worker(rank: int, world_size: int, args: argparse.Namespace, device_ids: Sequence[int]) -> None:
    use_ddp = ddp_available(world_size)
    if use_ddp:
        setup_ddp(rank, world_size)

    device = torch.device("cuda", rank) if torch.cuda.is_available() and device_ids else torch.device("cpu")

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
    elif world_size > 1 and len(device_ids) > 1:
        corrector = torch.nn.DataParallel(corrector)

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
            history = history.to(device, non_blocking=True) if history is not None else None
            optim.zero_grad()
            a_corr = corrector(z_real, z_pred, a_plan, mismatch_history=history)
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
    parser.add_argument("--gpus", type=str, default=_EARLY_GPUS or "all", help="GPU selection: all, N, or comma list")
    args = parser.parse_args()

    device_ids = [] if args.device.startswith("cpu") else _resolve_device_ids(args.gpus)
    requested = args.gpus if args.gpus is not None else "all"
    print(
        f"[GPU CONFIG] Requested: {requested}; visible devices: {torch.cuda.device_count()} | "
        f"selected ids: {device_ids}"
    )

    world_size = len(device_ids) if device_ids else 1
    if world_size > 1 and not ddp_available(world_size):
        print("WARNING: DDP requested but unavailable. Falling back to DataParallel if possible.")
    env_world_size = int(os.environ.get("WORLD_SIZE", world_size))

    if env_world_size > 1:
        # Launched via torchrun
        train_worker(int(os.environ.get("LOCAL_RANK", 0)), env_world_size, args, device_ids)
    elif world_size > 1:
        torch.multiprocessing.spawn(train_worker, args=(world_size, args, device_ids), nprocs=world_size, join=True)
    else:
        train_worker(0, 1, args, device_ids)


if __name__ == "__main__":
    main()
