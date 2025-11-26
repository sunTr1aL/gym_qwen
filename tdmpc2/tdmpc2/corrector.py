"""Corrector architectures for speculative TD-MPC2 execution.

This module implements two residual action correctors:

* :class:`TwoTowerCorrector` – a gated two-tower MLP that fuses features from the
  real latent, predicted latent, their difference, and the planned action.
* :class:`TemporalTransformerCorrector` – a lightweight Transformer encoder over a
  short history of mismatch features.

Both correctors expose the same interface and return a corrected action
``a_corr = a_plan + delta_a``. The helper :func:`build_corrector_from_cfg` builds the
appropriate instance from a config object.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn

__all__ = [
    "CorrectorConfig",
    "BaseCorrector",
    "TwoTowerCorrector",
    "TemporalTransformerCorrector",
    "Corrector",
    "build_corrector_from_cfg",
    "corrector_loss",
]


@dataclass
class CorrectorConfig:
    latent_dim: int
    act_dim: int
    hidden_dim: int = 256
    num_layers: int = 2
    activation: str = "silu"
    corrector_type: str = "two_tower"
    history_len: int = 4
    transformer_d_model: int = 128
    transformer_num_layers: int = 2
    transformer_num_heads: int = 4
    transformer_ff_dim: int = 256
    tanh_output: bool = True


class BaseCorrector(nn.Module):
    """Base interface for speculative action correctors."""

    def forward(
        self,
        z_real: torch.Tensor,
        z_pred: torch.Tensor,
        a_plan: torch.Tensor,
        mismatch_history: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        raise NotImplementedError


def _make_mlp(input_dim: int, hidden_dim: int, out_dim: int, num_layers: int, activation: nn.Module):
    layers = []
    last_dim = input_dim
    for _ in range(max(num_layers - 1, 0)):
        layers.append(nn.Linear(last_dim, hidden_dim))
        layers.append(activation())
        last_dim = hidden_dim
    layers.append(nn.Linear(last_dim, out_dim))
    return nn.Sequential(*layers)


class TwoTowerCorrector(BaseCorrector):
    """Gated two-tower MLP corrector.

    Consumes ``z_real``, ``z_pred``, and ``a_plan`` to predict a residual Δa that nudges the
    planned action toward what the TD-MPC2 teacher would output when replanning.
    """

    def __init__(
        self,
        latent_dim: int,
        act_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 2,
        activation: Optional[nn.Module] = None,
        tanh_output: bool = True,
    ) -> None:
        super().__init__()
        act = activation or nn.SiLU
        self.tanh_output = tanh_output
        self.mlp_real = _make_mlp(latent_dim, hidden_dim, hidden_dim, num_layers, act)
        self.mlp_pred = _make_mlp(latent_dim, hidden_dim, hidden_dim, num_layers, act)
        self.mlp_delta = _make_mlp(latent_dim, hidden_dim, hidden_dim, num_layers, act)
        self.mlp_a = _make_mlp(act_dim, hidden_dim, hidden_dim, num_layers, act)
        fusion_in = hidden_dim * 4
        self.mlp_u = _make_mlp(fusion_in, hidden_dim, hidden_dim, num_layers, act)
        self.mlp_g = _make_mlp(fusion_in, hidden_dim, hidden_dim, num_layers, act)
        self.out = nn.Linear(hidden_dim, act_dim)

    def forward(
        self,
        z_real: torch.Tensor,
        z_pred: torch.Tensor,
        a_plan: torch.Tensor,
        mismatch_history: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        z_real, z_pred, a_plan = self._ensure_batch(z_real, z_pred, a_plan)
        delta = z_real - z_pred
        h_real = self.mlp_real(z_real)
        h_pred = self.mlp_pred(z_pred)
        h_delta = self.mlp_delta(delta)
        h_a = self.mlp_a(a_plan)
        h = torch.cat([h_real, h_pred, h_delta, h_a], dim=-1)
        u = self.mlp_u(h)
        g = torch.sigmoid(self.mlp_g(h))
        fused = g * u + (1 - g) * h_a
        delta_a = self.out(fused)
        if self.tanh_output:
            delta_a = torch.tanh(delta_a)
        return a_plan + delta_a

    @staticmethod
    def _ensure_batch(z_real: torch.Tensor, z_pred: torch.Tensor, a_plan: torch.Tensor):
        if z_real.ndim == 1:
            z_real = z_real.unsqueeze(0)
        if z_pred.ndim == 1:
            z_pred = z_pred.unsqueeze(0)
        if a_plan.ndim == 1:
            a_plan = a_plan.unsqueeze(0)
        return z_real, z_pred, a_plan


class TemporalTransformerCorrector(BaseCorrector):
    """Temporal corrector using a Transformer encoder over mismatch history."""

    def __init__(
        self,
        latent_dim: int,
        act_dim: int,
        history_len: int = 4,
        hidden_dim: int = 256,
        num_layers: int = 2,
        activation: Optional[nn.Module] = None,
        d_model: int = 128,
        num_heads: int = 4,
        ff_dim: int = 256,
        tanh_output: bool = True,
    ) -> None:
        super().__init__()
        act = activation or nn.SiLU
        self.tanh_output = tanh_output
        self.history_len = history_len
        self.feat_dim = 3 * latent_dim + act_dim
        self.input_proj = nn.Linear(self.feat_dim, d_model)
        self.pos_emb = nn.Parameter(torch.zeros(history_len, d_model))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=num_heads, dim_feedforward=ff_dim, activation="gelu"
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.mlp_a = _make_mlp(act_dim, hidden_dim, hidden_dim, num_layers, act)
        fusion_in = hidden_dim + d_model
        self.mlp_u = _make_mlp(fusion_in, hidden_dim, hidden_dim, num_layers, act)
        self.mlp_g = _make_mlp(fusion_in, hidden_dim, hidden_dim, num_layers, act)
        self.out = nn.Linear(hidden_dim, act_dim)

    def forward(
        self,
        z_real: torch.Tensor,
        z_pred: torch.Tensor,
        a_plan: torch.Tensor,
        mismatch_history: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        z_real, z_pred, a_plan = TwoTowerCorrector._ensure_batch(z_real, z_pred, a_plan)
        if mismatch_history is None:
            # create a dummy history with the current step as the only entry
            feat = torch.cat([z_real, z_pred, z_real - z_pred, a_plan], dim=-1)
            mismatch_history = feat.unsqueeze(1).repeat(1, self.history_len, 1)
        if mismatch_history.ndim == 2:
            mismatch_history = mismatch_history.unsqueeze(0)

        seq = self.input_proj(mismatch_history) + self.pos_emb.unsqueeze(0)
        seq = seq.transpose(0, 1)  # [K, B, d_model]
        h_seq = self.encoder(seq).transpose(0, 1)  # [B, K, d_model]
        h_ctx = h_seq[:, 0]  # use most recent position
        h_a = self.mlp_a(a_plan)
        h = torch.cat([h_ctx, h_a], dim=-1)
        u = self.mlp_u(h)
        g = torch.sigmoid(self.mlp_g(h))
        fused = g * u + (1 - g) * h_a
        delta_a = self.out(fused)
        if self.tanh_output:
            delta_a = torch.tanh(delta_a)
        return a_plan + delta_a


class Corrector(TwoTowerCorrector):
    """Backwards-compatible default corrector alias.

    This wrapper preserves the historical ``Corrector`` API that exposed a simple
    ``model`` attribute and ``loss_fn`` while reusing :class:`TwoTowerCorrector`
    as the underlying implementation. When ``self.model`` is set, the forward
    pass mirrors the legacy behavior by concatenating features and applying the
    provided module to predict the residual.
    """

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        latent_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 2,
        tanh_output: bool = True,
        activation: Optional[nn.Module] = None,
    ) -> None:
        super().__init__(
            latent_dim=latent_dim,
            act_dim=act_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            activation=activation,
            tanh_output=tanh_output,
        )
        self.model: Optional[nn.Module] = None
        self.obs_dim = obs_dim

    def forward(
        self,
        z_real: torch.Tensor,
        z_pred: torch.Tensor,
        a_plan: torch.Tensor,
        mismatch_history: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.model is None:
            return super().forward(z_real, z_pred, a_plan, mismatch_history)

        z_real, z_pred, a_plan = self._ensure_batch(z_real, z_pred, a_plan)
        feat = torch.cat([z_real, z_pred, z_real - z_pred, a_plan], dim=-1)
        delta_a = self.model(feat)
        if self.tanh_output:
            delta_a = torch.tanh(delta_a)
        return a_plan + delta_a

    def loss_fn(
        self,
        a_corr: torch.Tensor,
        a_teacher: torch.Tensor,
        *,
        reg_lambda: float = 0.0,
        a_spec: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        plan = a_spec if a_spec is not None else a_corr.detach()
        return corrector_loss(a_corr, a_teacher, plan, reg_lambda=reg_lambda)


def build_corrector_from_cfg(cfg, latent_dim: int, act_dim: int, device: Optional[torch.device] = None) -> BaseCorrector:
    """Instantiate a corrector from configuration."""

    act = nn.SiLU
    corrector_type = getattr(cfg, "corrector_type", "two_tower")
    tanh_output = bool(getattr(cfg, "corrector_tanh_output", True))
    if corrector_type == "temporal":
        corrector: BaseCorrector = TemporalTransformerCorrector(
            latent_dim=latent_dim,
            act_dim=act_dim,
            history_len=int(getattr(cfg, "spec_history_len", 4)),
            hidden_dim=int(getattr(cfg, "corrector_hidden_dim", 256)),
            num_layers=int(getattr(cfg, "corrector_layers", 2)),
            activation=act,
            d_model=int(getattr(cfg, "transformer_d_model", 128)),
            num_heads=int(getattr(cfg, "transformer_heads", 4)),
            ff_dim=int(getattr(cfg, "transformer_ff_dim", 256)),
            tanh_output=tanh_output,
        )
    else:
        corrector = TwoTowerCorrector(
            latent_dim=latent_dim,
            act_dim=act_dim,
            hidden_dim=int(getattr(cfg, "corrector_hidden_dim", 256)),
            num_layers=int(getattr(cfg, "corrector_layers", 2)),
            activation=act,
            tanh_output=tanh_output,
        )
    if device is not None:
        corrector = corrector.to(device)
    return corrector


def corrector_loss(a_corr: torch.Tensor, a_teacher: torch.Tensor, a_plan: torch.Tensor, reg_lambda: float = 0.0) -> torch.Tensor:
    """Compute distillation loss with optional residual regularization."""

    mse = torch.nn.functional.mse_loss(a_corr, a_teacher)
    if reg_lambda > 0:
        delta_a = a_corr - a_plan
        mse = mse + reg_lambda * (delta_a.pow(2).mean())
    return mse

