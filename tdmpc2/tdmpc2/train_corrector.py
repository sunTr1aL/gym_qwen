import argparse
from pathlib import Path
from typing import Dict, List

import torch
from torch.utils.data import DataLoader, Dataset

from tdmpc2.corrector import build_corrector_from_cfg, corrector_loss


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
    args = parser.parse_args()

    device = torch.device(args.device)
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
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    corrector = build_corrector_from_cfg(cfg, latent_dim=latent_dim, act_dim=act_dim, device=device)
    optim = torch.optim.Adam(corrector.parameters(), lr=args.lr)

    for epoch in range(args.epochs):
        total_loss = 0.0
        count = 0
        for batch in loader:
            z_real, z_pred, a_plan, a_teacher, history = batch
            optim.zero_grad()
            a_corr = corrector(z_real, z_pred, a_plan, mismatch_history=history)
            loss = corrector_loss(a_corr, a_teacher, a_plan, reg_lambda=args.reg_lambda)
            loss.backward()
            optim.step()
            total_loss += loss.item() * z_real.shape[0]
            count += z_real.shape[0]
        avg_loss = total_loss / max(count, 1)
        print(f"Epoch {epoch+1}/{args.epochs} - loss: {avg_loss:.4f}")

    torch.save({"corrector": corrector.state_dict(), "corrector_type": args.corrector_type}, args.save_path)
    print(f"Saved trained corrector to {args.save_path}")


if __name__ == "__main__":
    main()
