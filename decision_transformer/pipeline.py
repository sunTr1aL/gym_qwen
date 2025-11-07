from typing import List, Tuple

import torch
import torch.nn as nn

from .model import Block, Qwen3Block, RMSNorm, RotaryEmbedding


def _distribute_layers(num_layers: int, num_stages: int) -> List[int]:
    if num_stages < 1:
        raise ValueError("num_stages must be >= 1.")
    if num_layers < 0:
        raise ValueError("num_layers must be >= 0.")
    base = num_layers // num_stages
    remainder = num_layers % num_stages
    dist = [base for _ in range(num_stages)]
    if remainder:
        dist[-1] += remainder
    return dist


# ---------------------------------------------------------------------------
# Decision Transformer stages
# ---------------------------------------------------------------------------


class _DTStageBase(nn.Module):
    def __init__(
        self,
        h_dim: int,
        context_len: int,
        n_heads: int,
        drop_p: float,
        num_blocks: int,
    ) -> None:
        super().__init__()
        max_T = 3 * context_len
        self.blocks = nn.ModuleList(
            Block(h_dim, max_T, n_heads, drop_p) for _ in range(num_blocks)
        )
        self.h_dim = h_dim

    def _run_blocks(self, h: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            h = block(h)
        return h


class DTStageInput(_DTStageBase):
    def __init__(
        self,
        state_dim: int,
        act_dim: int,
        h_dim: int,
        context_len: int,
        n_heads: int,
        drop_p: float,
        max_timestep: int,
        num_blocks: int,
    ) -> None:
        super().__init__(h_dim, context_len, n_heads, drop_p, num_blocks)
        self.embed_ln = nn.LayerNorm(h_dim)
        self.embed_timestep = nn.Embedding(max_timestep, h_dim)
        self.embed_rtg = nn.Linear(1, h_dim)
        self.embed_state = nn.Linear(state_dim, h_dim)
        self.embed_action = nn.Linear(act_dim, h_dim)

    def forward(
        self,
        inputs: Tuple[
            torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
        ],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        timesteps, states, actions, returns_to_go, traj_mask = inputs
        B, T, _ = states.shape

        time_embeddings = self.embed_timestep(timesteps)
        state_embeddings = self.embed_state(states) + time_embeddings
        action_embeddings = self.embed_action(actions) + time_embeddings
        returns_embeddings = self.embed_rtg(returns_to_go) + time_embeddings

        h = torch.stack(
            (returns_embeddings, state_embeddings, action_embeddings), dim=1
        ).permute(0, 2, 1, 3).reshape(B, 3 * T, self.h_dim)

        h = self.embed_ln(h)
        h = self._run_blocks(h)

        return h, traj_mask


class DTStageMiddle(_DTStageBase):
    def forward(
        self, inputs: Tuple[torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        h, traj_mask = inputs
        h = self._run_blocks(h)
        return h, traj_mask


class DTStageOutput(_DTStageBase):
    def __init__(
        self,
        h_dim: int,
        state_dim: int,
        act_dim: int,
        context_len: int,
        n_heads: int,
        drop_p: float,
        num_blocks: int,
        use_action_tanh: bool,
    ) -> None:
        super().__init__(h_dim, context_len, n_heads, drop_p, num_blocks)
        self.predict_rtg = nn.Linear(h_dim, 1)
        self.predict_state = nn.Linear(h_dim, state_dim)
        head_layers: List[nn.Module] = [nn.Linear(h_dim, act_dim)]
        if use_action_tanh:
            head_layers.append(nn.Tanh())
        self.predict_action = nn.Sequential(*head_layers)

    def forward(
        self, inputs: Tuple[torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        h, traj_mask = inputs
        h = self._run_blocks(h)

        B = h.size(0)
        T = traj_mask.size(1)
        h = h.reshape(B, T, 3, self.h_dim).permute(0, 2, 1, 3)

        return_preds = self.predict_rtg(h[:, 2])
        state_preds = self.predict_state(h[:, 2])
        action_preds = self.predict_action(h[:, 1])
        return state_preds, action_preds, return_preds, traj_mask


def build_dt_pipeline_stages(
    *,
    state_dim: int,
    act_dim: int,
    context_len: int,
    n_blocks: int,
    h_dim: int,
    n_heads: int,
    drop_p: float,
    max_timestep: int,
    use_action_tanh: bool,
    num_stages: int,
) -> Tuple[List[nn.Module], List[int]]:
    layers_per_stage = _distribute_layers(n_blocks, num_stages)
    stages: List[nn.Module] = []
    for idx, num_blocks in enumerate(layers_per_stage):
        if idx == 0:
            stages.append(
                DTStageInput(
                    state_dim=state_dim,
                    act_dim=act_dim,
                    h_dim=h_dim,
                    context_len=context_len,
                    n_heads=n_heads,
                    drop_p=drop_p,
                    max_timestep=max_timestep,
                    num_blocks=num_blocks,
                )
            )
        elif idx == num_stages - 1:
            stages.append(
                DTStageOutput(
                    h_dim=h_dim,
                    state_dim=state_dim,
                    act_dim=act_dim,
                    context_len=context_len,
                    n_heads=n_heads,
                    drop_p=drop_p,
                    num_blocks=num_blocks,
                    use_action_tanh=use_action_tanh,
                )
            )
        else:
            stages.append(
                DTStageMiddle(
                    h_dim=h_dim,
                    context_len=context_len,
                    n_heads=n_heads,
                    drop_p=drop_p,
                    num_blocks=num_blocks,
                )
            )
    return stages, layers_per_stage


# ---------------------------------------------------------------------------
# Qwen3 stages
# ---------------------------------------------------------------------------


class _QwenStageBase(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        context_len: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        attn_dropout: float,
        mlp_ratio: float,
        rope_theta: float,
        num_blocks: int,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.blocks = nn.ModuleList(
            _PipelineQwenBlock(
                hidden_size=hidden_size,
                num_heads=num_heads,
                num_kv_heads=num_kv_heads,
                head_dim=head_dim,
                context_len=context_len,
                rope_theta=rope_theta,
                mlp_ratio=mlp_ratio,
                attn_dropout=attn_dropout,
            )
            for _ in range(num_blocks)
        )

    def _run_blocks(self, h: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
        for blk in self.blocks:
            h = blk(h, attn_mask)
        return h


class _PipelineQwenBlock(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        context_len: int,
        rope_theta: float,
        mlp_ratio: float,
        attn_dropout: float,
    ) -> None:
        super().__init__()
        rope = RotaryEmbedding(
            head_dim=head_dim,
            max_position_embeddings=3 * context_len,
            rope_theta=rope_theta,
        )
        self.inner = Qwen3Block(
            hidden_size=hidden_size,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            mlp_ratio=mlp_ratio,
            attn_dropout=attn_dropout,
            bias=False,
            rope=rope,
        )

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
        return self.inner(x, attn_mask)


class QwenStageInput(_QwenStageBase):
    def __init__(
        self,
        state_dim: int,
        act_dim: int,
        context_len: int,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        attn_dropout: float,
        drop_p: float,
        rope_theta: float,
        max_timestep: int,
        use_action_tanh: bool,
        num_blocks: int,
    ) -> None:
        super().__init__(
            hidden_size=hidden_size,
            context_len=context_len,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            attn_dropout=attn_dropout,
            mlp_ratio=3.0,
            rope_theta=rope_theta,
            num_blocks=num_blocks,
        )
        self.context_len = context_len
        self.embed_ln = RMSNorm(hidden_size)
        self.embed_timestep = nn.Embedding(max_timestep, hidden_size)
        self.embed_rtg = nn.Linear(1, hidden_size)
        self.embed_state = nn.Linear(state_dim, hidden_size)
        self.embed_action = nn.Linear(act_dim, hidden_size)
        self.drop = nn.Dropout(drop_p)
        max_T = 3 * context_len
        mask = torch.tril(torch.ones((max_T, max_T)))
        self.register_buffer("causal_mask", mask.view(1, 1, max_T, max_T))

    def _causal(self, seq_len: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        mask = self.causal_mask[..., :seq_len, :seq_len].to(device=device)
        return (mask == 0).to(dtype=dtype) * torch.finfo(dtype).min

    def forward(
        self,
        inputs: Tuple[
            torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
        ],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        timesteps, states, actions, returns_to_go, traj_mask = inputs
        B, T, _ = states.shape

        time_embeddings = self.embed_timestep(timesteps)
        state_embeddings = self.embed_state(states) + time_embeddings
        action_embeddings = self.embed_action(actions) + time_embeddings
        returns_embeddings = self.embed_rtg(returns_to_go) + time_embeddings

        h = torch.stack(
            (returns_embeddings, state_embeddings, action_embeddings), dim=1
        ).permute(0, 2, 1, 3).reshape(B, 3 * T, self.hidden_size)

        h = self.embed_ln(h)
        h = self.drop(h)

        attn_mask = self._causal(h.size(1), device=h.device, dtype=h.dtype)
        h = self._run_blocks(h, attn_mask)

        return h, attn_mask, traj_mask


class QwenStageMiddle(_QwenStageBase):
    def forward(
        self, inputs: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h, attn_mask, traj_mask = inputs
        h = self._run_blocks(h, attn_mask)
        return h, attn_mask, traj_mask


class QwenStageOutput(_QwenStageBase):
    def __init__(
        self,
        state_dim: int,
        act_dim: int,
        context_len: int,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        attn_dropout: float,
        rope_theta: float,
        drop_p: float,
        use_action_tanh: bool,
        num_blocks: int,
    ) -> None:
        super().__init__(
            hidden_size=hidden_size,
            context_len=context_len,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            attn_dropout=attn_dropout,
            mlp_ratio=3.0,
            rope_theta=rope_theta,
            num_blocks=num_blocks,
        )
        self.predict_rtg = nn.Linear(hidden_size, 1)
        self.predict_state = nn.Linear(hidden_size, state_dim)
        head_layers: List[nn.Module] = [nn.Linear(hidden_size, act_dim)]
        if use_action_tanh:
            head_layers.append(nn.Tanh())
        self.predict_action = nn.Sequential(*head_layers)

    def forward(
        self, inputs: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        h, attn_mask, traj_mask = inputs
        h = self._run_blocks(h, attn_mask)

        B = h.size(0)
        T = traj_mask.size(1)
        h = h.view(B, T, 3, self.hidden_size).permute(0, 2, 1, 3)

        return_preds = self.predict_rtg(h[:, 2])
        state_preds = self.predict_state(h[:, 2])
        action_preds = self.predict_action(h[:, 1])
        return state_preds, action_preds, return_preds, traj_mask


def build_qwen3_pipeline_stages(
    *,
    state_dim: int,
    act_dim: int,
    context_len: int,
    n_layers: int,
    hidden_size: int,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    attn_dropout: float,
    drop_p: float,
    rope_theta: float,
    max_timestep: int,
    use_action_tanh: bool,
    num_stages: int,
) -> Tuple[List[nn.Module], List[int]]:
    layers_per_stage = _distribute_layers(n_layers, num_stages)
    stages: List[nn.Module] = []
    for idx, num_blocks in enumerate(layers_per_stage):
        if idx == 0:
            stages.append(
                QwenStageInput(
                    state_dim=state_dim,
                    act_dim=act_dim,
                    context_len=context_len,
                    hidden_size=hidden_size,
                    num_heads=num_heads,
                    num_kv_heads=num_kv_heads,
                    head_dim=head_dim,
                    attn_dropout=attn_dropout,
                    drop_p=drop_p,
                    rope_theta=rope_theta,
                    max_timestep=max_timestep,
                    use_action_tanh=use_action_tanh,
                    num_blocks=num_blocks,
                )
            )
        elif idx == num_stages - 1:
            stages.append(
                QwenStageOutput(
                    state_dim=state_dim,
                    act_dim=act_dim,
                    context_len=context_len,
                    hidden_size=hidden_size,
                    num_heads=num_heads,
                    num_kv_heads=num_kv_heads,
                    head_dim=head_dim,
                    attn_dropout=attn_dropout,
                    rope_theta=rope_theta,
                    drop_p=drop_p,
                    use_action_tanh=use_action_tanh,
                    num_blocks=num_blocks,
                )
            )
        else:
            stages.append(
                QwenStageMiddle(
                    hidden_size=hidden_size,
                    context_len=context_len,
                    num_heads=num_heads,
                    num_kv_heads=num_kv_heads,
                    head_dim=head_dim,
                    attn_dropout=attn_dropout,
                    mlp_ratio=3.0,
                    rope_theta=rope_theta,
                    num_blocks=num_blocks,
                )
            )
    return stages, layers_per_stage
