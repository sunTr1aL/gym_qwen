import argparse
import copy
import csv
import datetime
import json
import os
import time
from pathlib import Path
from typing import Dict, List

import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

from tdmpc2.corrector import build_corrector_from_cfg, corrector_loss
from tdmpc2.launch import cleanup_ddp, ddp_available, launch, setup_ddp, wrap_dataparallel
from tdmpc2.utils_ckpt import list_pretrained_checkpoints


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

    for meta_key in ("model_id", "model_name", "model_size"):
        state.pop(meta_key, None)

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


def discover_dataset_ids(data_dir: str) -> List[str]:
    ids: List[str] = []
    for path in Path(data_dir).glob("corrector_data_*.pt"):
        name = path.stem.replace("corrector_data_", "")
        if not name:
            continue
        if name not in ids:
            ids.append(name)
    return sorted(ids)


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
    history = {"epoch": [], "train_loss": [], "train_mse": [], "train_delta_norm": []}
    for epoch in range(args.epochs):
        if sampler is not None:
            sampler.set_epoch(epoch)
        total_loss = 0.0
        total_mse = 0.0
        total_delta_norm = 0.0
        count = 0
        epoch_start = time.time()
        for batch in loader:
            z_real, z_pred, a_plan, a_teacher, hist_feats = batch
            z_real = z_real.to(device, non_blocking=True)
            z_pred = z_pred.to(device, non_blocking=True)
            a_plan = a_plan.to(device, non_blocking=True)
            a_teacher = a_teacher.to(device, non_blocking=True)
            if args.corrector_type == "temporal":
                if hist_feats.numel() == 0:
                    raise ValueError(
                        "Temporal corrector requires 'history_feats' in the dataset; received empty history."
                    )
                mismatch_history = hist_feats.to(device, non_blocking=True)
            else:
                mismatch_history = None
            optim.zero_grad()
            a_corr = corrector(z_real, z_pred, a_plan, mismatch_history=mismatch_history)
            mse = torch.nn.functional.mse_loss(a_corr, a_teacher)
            delta_norm = (a_corr - a_plan).norm(dim=-1).mean()
            loss = corrector_loss(a_corr, a_teacher, a_plan, reg_lambda=args.reg_lambda)
            loss.backward()
            optim.step()
            total_loss += loss.item() * z_real.shape[0]
            total_mse += mse.item() * z_real.shape[0]
            total_delta_norm += delta_norm.item() * z_real.shape[0]
            count += z_real.shape[0]
        if rank == 0:
            avg_loss = total_loss / max(count, 1)
            avg_mse = total_mse / max(count, 1)
            avg_norm = total_delta_norm / max(count, 1)
            history["epoch"].append(epoch + 1)
            history["train_loss"].append(avg_loss)
            history["train_mse"].append(avg_mse)
            history["train_delta_norm"].append(avg_norm)
            elapsed = time.time() - epoch_start
            samples_per_sec = count / max(elapsed, 1e-6)
            print(
                f"Epoch {epoch+1}/{args.epochs} - loss: {avg_loss:.4f} - mse: {avg_mse:.4f} - "
                f"delta_norm: {avg_norm:.4f} - "
                f"samples/sec: {samples_per_sec:.1f}"
            )

    if rank == 0:
        state = corrector.module.state_dict() if hasattr(corrector, "module") else corrector.state_dict()
        torch.save(
            {
                "corrector": state,
                "corrector_type": args.corrector_type,
                "model_id": getattr(args, "model_id", None),
                "model_name": getattr(args, "model_name", None),
                "model_size": getattr(args, "model_size", None),
            },
            args.save_path,
        )
        os.makedirs(args.results_dir, exist_ok=True)
        ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        run_id_parts = [f"{args.corrector_type}", f"bs{args.batch_size}", f"lr{args.lr}"]
        if getattr(args, "seed", None) is not None:
            run_id_parts.append(f"seed{args.seed}")
        run_id = "_".join(run_id_parts)
        base = os.path.join(args.results_dir, f"{run_id}_{ts}")
        meta = {
            "corrector_type": args.corrector_type,
            "latent_dim": latent_dim,
            "act_dim": act_dim,
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "lr": args.lr,
            "lambda_reg": args.reg_lambda,
            "filter_min_distance": args.filter_min_distance,
            "dataset_path": args.data,
            "model_id": getattr(args, "model_id", None),
            "model_name": getattr(args, "model_name", None),
            "model_size": getattr(args, "model_size", None),
            "seed": args.seed,
        }
        with open(base + "_metrics.json", "w", encoding="utf-8") as f:
            json.dump({"meta": meta, "history": history}, f, indent=2)
        with open(base + "_metrics.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["epoch", "train_loss", "train_mse", "train_delta_norm"])
            for i in range(len(history["epoch"])):
                writer.writerow(
                    [
                        history["epoch"][i],
                        history["train_loss"][i],
                        history["train_mse"][i],
                        history["train_delta_norm"][i],
                    ]
                )
        total_time = time.time() - global_start
        print(
            f"Saved trained corrector to {args.save_path} after {args.epochs} epochs "
            f"({len(dataset) * args.epochs / max(total_time, 1e-6):.1f} samples/sec)."
        )

    if use_ddp:
        cleanup_ddp()


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a speculative action corrector offline.")
    parser.add_argument("--data", type=str, required=False, help="Path to saved corrector buffer (.pt) or directory of buffers")
    parser.add_argument("--data_dir", type=str, default="data", help="Directory containing per-model datasets")
    parser.add_argument("--checkpoint_dir", type=str, default="tdmpc2_pretrained", help="Directory containing pretrained TD-MPC2 checkpoints")
    parser.add_argument("--model_id", type=str, default=None, help="Model id (checkpoint stem) to train for")
    parser.add_argument("--model_size", type=str, default=None, help="Filter checkpoints by size token (e.g., 5m)")
    parser.add_argument("--all_models", action="store_true", help="Train correctors for every checkpoint discovered")
    parser.add_argument("--all_model_sizes", action="store_true", help="Alias for --all_models")
    parser.add_argument(
        "--save_path", type=str, default="corrector.pth", help="Output path for trained weights (single run)",
    )
    parser.add_argument("--corrector_dir", type=str, default="correctors", help="Directory to store per-size correctors")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--reg_lambda", type=float, default=0.0, help="Residual L2 regularization weight")
    parser.add_argument("--filter_min_distance", type=float, default=0.0, help="Minimum distance to keep a sample")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--results_dir",
        type=str,
        default="results/corrector_train",
        help="Directory where training metrics will be saved (JSON + CSV).",
    )
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--corrector_type",
        type=str,
        default="two_tower",
        choices=["two_tower", "temporal", "both"],
    )
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--history_len", type=int, default=4)
    parser.add_argument(
        "--gpus", type=str, default="1", help="GPU selection: 'all', N, or comma-separated list"
    )
    parser.add_argument(
        "--exclude_pattern",
        action="append",
        help="Optional substring(s) to skip when discovering checkpoints",
    )
    args = parser.parse_args()

    if args.all_model_sizes:
        args.all_models = True

    target_models: List[tuple[str, Dict[str, str] | None]]
    ckpt_infos: Dict[str, Dict[str, str]] = {}
    if args.all_models or args.model_id or args.model_size:
        ckpt_infos = list_pretrained_checkpoints(
            args.checkpoint_dir, model_size_filter=args.model_size
        )
        if not ckpt_infos:
            raise ValueError(f"No checkpoints found in {args.checkpoint_dir} matching filters.")
        if args.model_id:
            if args.model_id not in ckpt_infos:
                raise ValueError(
                    f"Model id '{args.model_id}' not found in {args.checkpoint_dir}. Available: {list(ckpt_infos.keys())}"
                )
            target_models = [(args.model_id, ckpt_infos[args.model_id])]
        else:
            target_models = list(ckpt_infos.items())
    elif args.data is None:
        ids = discover_dataset_ids(args.data_dir)
        if not ids:
            raise ValueError("No datasets found; specify --data or ensure data_dir has corrector_data_<model_id>.pt")
        target_models = [(mid, None) for mid in ids]
    else:
        target_models = [(Path(args.data).stem.replace("corrector_data_", "") or None, None)]

    target_types = ["two_tower", "temporal"] if args.corrector_type == "both" else [args.corrector_type]

    for model_id, info in target_models:
        model_name = info.get("model_name") if info else None
        model_size = info.get("model_size") if info else None
        for corr_type in target_types:
            run_args = copy.deepcopy(args)
            run_args.corrector_type = corr_type
            run_args.model_id = model_id
            run_args.model_name = model_name
            run_args.model_size = model_size
            if args.data is None:
                if model_id is None:
                    raise ValueError("Dataset path (--data) required when model_id is not provided.")
                run_args.data = os.path.join(args.data_dir, f"corrector_data_{model_id}.pt")
            if args.save_path == "corrector.pth" or args.corrector_type == "both" or args.all_models or not args.model_id:
                os.makedirs(args.corrector_dir, exist_ok=True)
                suffix = f"_{model_id}" if model_id else ""
                run_args.save_path = os.path.join(
                    args.corrector_dir, f"corrector{suffix}_{corr_type}.pth"
                )
            name_size = f" ({model_name}-{model_size})" if model_name or model_size else ""
            print(
                f"Training corrector type={corr_type} for model_id={model_id or 'dataset'}{name_size} using {run_args.data}"
            )
            launch(run_args, train_worker, use_ddp=True, allow_dataparallel=True)




if __name__ == "__main__":
    main()
