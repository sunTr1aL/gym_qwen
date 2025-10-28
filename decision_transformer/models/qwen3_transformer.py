
import math
from dataclasses import dataclass
from typing import Optional, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


def rms_norm(x, weight, eps: float = 1e-6):
    # x: (B, T, C) or (N, C)
    return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps) * weight


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return rms_norm(x, self.weight, self.eps)


def rotate_half(x):
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


class RotaryEmbedding(nn.Module):
    """
    RoPE with base theta and precomputed cos/sin cache.
    """
    def __init__(self, dim: int, base: float = 10000.0, max_position: int = 4096, device: Optional[torch.device] = None):
        super().__init__()
        self.dim = dim
        self.base = base
        self.max_position = max_position
        self.register_buffer("cos_cached", torch.empty(1, 1, max_position, dim), persistent=False)
        self.register_buffer("sin_cached", torch.empty(1, 1, max_position, dim), persistent=False)
        self._build_cache(device=device)

    @torch.no_grad()
    def _build_cache(self, device=None):
        device = device or self.cos_cached.device
        # frequencies
        theta = self.base ** (torch.arange(0, self.dim, 2, dtype=torch.float32, device=device) / self.dim)
        # positions 0..T-1
        t = torch.arange(self.max_position, dtype=torch.float32, device=device)[:, None]
        freqs = t / theta[None, :]
        emb = torch.cat([freqs, freqs], dim=-1)  # (max_pos, dim)
        self.cos_cached = emb.cos()[None, None, ...]  # (1,1,T,dim)
        self.sin_cached = emb.sin()[None, None, ...]

    def forward(self, x, positions: torch.Tensor):
        """
        x: (..., T, dim)
        positions: (T,) or (B, T) integer indices in [0, max_position)
        """
        # Broadcast cos/sin to x shape
        if positions.dim() == 1:
            cos = self.cos_cached[:, :, positions, :]
            sin = self.sin_cached[:, :, positions, :]
        else:
            # positions: (B, T)
            B, T = positions.shape
            cos = self.cos_cached[:, :, positions.reshape(B, T, 1).expand(B, T, self.dim), :]
            sin = self.sin_cached[:, :, positions.reshape(B, T, 1).expand(B, T, self.dim), :]
        return (x * cos) + (rotate_half(x) * sin)


@dataclass
class QwenBlockConfig:
    d_model: int = 2048
    n_heads: int = 16
    n_kv_heads: int = 8      # Grouped Query Attention; set equal to n_heads to disable sharing
    mlp_ratio: float = 5.4   # Qwen-style wider FFN
    rope_base: float = 10000.0
    max_position: int = 4096
    attn_dropout: float = 0.0
    resid_dropout: float = 0.0
    bias: bool = False
    use_flash: bool = True   # use scaled_dot_product_attention if available


class QwenSelfAttention(nn.Module):
    def __init__(self, cfg: QwenBlockConfig):
        super().__init__()
        self.cfg = cfg
        d = cfg.d_model
        self.n_heads = cfg.n_heads
        self.n_kv_heads = cfg.n_kv_heads
        self.head_dim = d // self.n_heads
        assert d % self.n_heads == 0, "d_model must be divisible by n_heads"
        assert self.n_heads % self.n_kv_heads == 0, "n_heads must be divisible by n_kv_heads"
        self.q_proj = nn.Linear(d, d, bias=cfg.bias)
        self.k_proj = nn.Linear(d, self.n_kv_heads * self.head_dim, bias=cfg.bias)
        self.v_proj = nn.Linear(d, self.n_kv_heads * self.head_dim, bias=cfg.bias)
        self.o_proj = nn.Linear(d, d, bias=cfg.bias)
        self.attn_drop = nn.Dropout(cfg.attn_dropdown if hasattr(cfg, "attn_dropdown") else cfg.attn_dropout)
        self.rope = RotaryEmbedding(self.head_dim, base=cfg.rope_base, max_position=cfg.max_position)

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor], positions: torch.Tensor):
        B, T, C = x.shape
        H, HK, D = self.n_heads, self.n_kv_heads, self.head_dim

        q = self.q_proj(x).view(B, T, H, D).transpose(1, 2)        # (B, H, T, D)
        k = self.k_proj(x).view(B, T, HK, D).transpose(1, 2)       # (B, HK, T, D)
        v = self.v_proj(x).view(B, T, HK, D).transpose(1, 2)       # (B, HK, T, D)

        # RoPE on q,k along the last dim using positions 0..T-1 within the window
        # positions: (B, T) or (T,)
        # Expand cos/sin to match (B, heads, T, D)
        # Apply rope per head
        # Prepare per-head positions
        if positions.dim() == 1:
            pos = positions
        else:
            # (B,T) -> (T,), assuming same positions across batch after padding slicing
            pos = positions[0]

        # broadcast rope caches to q and k shapes
        cos = self.rope.cos_cached[..., pos, :]  # (1,1,T,D)
        sin = self.rope.sin_cached[..., pos, :]
        # expand to heads
        cos_q = cos.expand(B, H, T, D)
        sin_q = sin.expand(B, H, T, D)
        cos_k = cos.expand(B, HK, T, D)
        sin_k = sin.expand(B, HK, T, D)

        q = (q * cos_q) + (rotate_half(q) * sin_q)
        k = (k * cos_k) + (rotate_half(k) * sin_k)

        # Grouped Query Attention: repeat K,V to match H
        if H != HK:
            # repeat_interleave along head dim
            repeat = H // HK
            k = k.repeat_interleave(repeat, dim=1)  # (B, H, T, D)
            v = v.repeat_interleave(repeat, dim=1)  # (B, H, T, D)

        # Scaled dot-product attention (uses FlashAttention kernel on PyTorch >=2.0 when possible)
        # Build causal mask and combine with provided attn_mask (B, T) where 1=keep, 0=mask
        causal = torch.ones((T, T), device=x.device, dtype=torch.bool).tril()
        if attn_mask is None:
            attn = F.scaled_dot_product_attention(q, k, v, attn_mask=~causal, dropout_p=0.0, is_causal=False)
        else:
            # attn_mask: (B, T) -> (B, 1, T) broadcast to heads; combine with causal
            am = attn_mask[:, None, None, :].bool()  # True for keep
            causal_mask = causal[None, None, :, :]   # (1,1,T,T)
            combined = am & causal_mask              # (B,1,T,T), True=keep
            attn = F.scaled_dot_product_attention(q, k, v, attn_mask=~combined, dropout_p=0.0, is_causal=False)

        out = attn.transpose(1, 2).contiguous().view(B, T, C)
        out = self.o_proj(out)
        return out


class SwiGLU(nn.Module):
    def __init__(self, dim_in: int, dim_out: int, multiple_of: int = 1, bias: bool = False):
        super().__init__()
        hidden = dim_out
        # Project to 2*hidden for gate
        self.w1 = nn.Linear(dim_in, hidden, bias=bias)
        self.w3 = nn.Linear(dim_in, hidden, bias=bias)
        self.w2 = nn.Linear(hidden, dim_in, bias=bias)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class QwenBlock(nn.Module):
    def __init__(self, cfg: QwenBlockConfig):
        super().__init__()
        self.ln1 = RMSNorm(cfg.d_model)
        self.attn = QwenSelfAttention(cfg)
        self.ln2 = RMSNorm(cfg.d_model)
        inner = int(cfg.mlp_ratio * cfg.d_model)
        self.mlp = SwiGLU(cfg.d_model, inner, bias=cfg.bias)

    def forward(self, x, attn_mask: Optional[torch.Tensor], positions: torch.Tensor):
        h = x + self.attn(self.ln1(x), attn_mask, positions)
        out = h + self.mlp(self.ln2(h))
        return out


@dataclass
class QwenTransformerConfig:
    d_model: int = 2048
    n_layer: int = 24
    n_heads: int = 16
    n_kv_heads: int = 8
    mlp_ratio: float = 5.4
    max_position: int = 4096
    rope_base: float = 10000.0
    bias: bool = False


class QwenTransformer(nn.Module):
    """
    Minimal Qwen3-style Transformer **without** token embeddings.
    Accepts `inputs_embeds` (B, T, d_model) and an optional attention_mask (B, T).
    Returns dict with key 'last_hidden_state' like HF models.
    """
    def __init__(self, config: QwenTransformerConfig):
        super().__init__()
        self.config = config
        block_cfg = QwenBlockConfig(
            d_model=config.d_model,
            n_heads=config.n_heads,
            n_kv_heads=config.n_kv_heads,
            mlp_ratio=config.mlp_ratio,
            max_position=config.max_position,
            rope_base=config.rope_base,
        )
        self.blocks = nn.ModuleList([QwenBlock(block_cfg) for _ in range(config.n_layer)])
        self.norm = RMSNorm(config.d_model)

    def forward(self, inputs_embeds: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        B, T, C = inputs_embeds.shape
        device = inputs_embeds.device
        # positions 0..T-1 (broadcastable)
        positions = torch.arange(T, device=device, dtype=torch.long)
        x = inputs_embeds
        for blk in self.blocks:
            x = blk(x, attention_mask, positions)
        x = self.norm(x)
        return {"last_hidden_state": x}
