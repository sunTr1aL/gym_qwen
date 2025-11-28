import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalSelfAttention(nn.Module):
	"""
	Causal self-attention with optional KV caching.

	Inputs/outputs use shape (B, T, C). Cache stores past keys/values per block to
	avoid recomputing attention during auto-regressive rollout (planning).
	"""

	def __init__(self, dim: int, num_heads: int, attn_dropout: float = 0.0, proj_dropout: float = 0.0):
		super().__init__()
		assert dim % num_heads == 0, "dim must be divisible by num_heads"
		self.dim = dim
		self.num_heads = num_heads
		self.head_dim = dim // num_heads
		self.scale = self.head_dim ** -0.5

		self.qkv = nn.Linear(dim, dim * 3, bias=False)
		self.proj = nn.Linear(dim, dim)
		self.attn_drop = nn.Dropout(attn_dropout)
		self.proj_drop = nn.Dropout(proj_dropout)

	def forward(
		self,
		x: torch.Tensor,
		cache: Optional[Dict[str, torch.Tensor]] = None,
		seq_start: int = 0,
	) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
		"""
		Args:
			x: float tensor of shape (B, T, C).
			cache: optional dict with keys 'k' and 'v', each shaped (B, H, T_cache, D).
			seq_start: absolute position index of the first token in `x` (used for masking).

		Returns:
			out: attended features, shape (B, T, C).
			new_cache: updated KV tensors detached from graph.
		"""
		B, T, C = x.shape
		qkv = self.qkv(x)  # (B, T, 3C)
		qkv = qkv.view(B, T, 3, self.num_heads, self.head_dim)
		q, k, v = qkv.unbind(dim=2)  # each (B, T, H, D)
		q = q.permute(0, 2, 1, 3)  # (B, H, T, D)
		k = k.permute(0, 2, 1, 3)
		v = v.permute(0, 2, 1, 3)

		if cache is not None:
			# Append new keys/values to cached history.
			k = torch.cat([cache["k"], k], dim=2)
			v = torch.cat([cache["v"], v], dim=2)

		# Build causal mask that accounts for cached tokens.
		k_len = k.shape[2]
		query_pos = torch.arange(seq_start, seq_start + T, device=x.device)
		key_pos = torch.arange(0, k_len, device=x.device)
		causal_mask = query_pos.unsqueeze(-1) >= key_pos.unsqueeze(0)  # (T, k_len)

		# Compute attention.
		q = q * self.scale
		attn = torch.einsum("bhqd,bhkd->bhqk", q, k)
		attn = attn.masked_fill(~causal_mask.unsqueeze(0).unsqueeze(0), float("-inf"))
		attn = F.softmax(attn, dim=-1)
		attn = self.attn_drop(attn)

		out = torch.einsum("bhqk,bhkd->bhqd", attn, v).contiguous()
		out = out.permute(0, 2, 1, 3).reshape(B, T, C)
		out = self.proj_drop(self.proj(out))

		new_cache = {
			"k": k.detach(),
			"v": v.detach(),
		}
		return out, new_cache


class TransformerBlock(nn.Module):
	"""
	Standard Transformer encoder block with pre-norm, residuals, and causal mask.
	"""

	def __init__(
		self,
		dim: int,
		num_heads: int,
		mlp_ratio: float = 4.0,
		attn_dropout: float = 0.0,
		dropout: float = 0.0,
	):
		super().__init__()
		self.norm1 = nn.LayerNorm(dim)
		self.attn = CausalSelfAttention(dim, num_heads, attn_dropout, proj_dropout=dropout)
		self.norm2 = nn.LayerNorm(dim)
		hidden_dim = int(dim * mlp_ratio)
		self.mlp = nn.Sequential(
			nn.Linear(dim, hidden_dim),
			nn.GELU(),
			nn.Dropout(dropout),
			nn.Linear(hidden_dim, dim),
			nn.Dropout(dropout),
		)

	def forward(
		self,
		x: torch.Tensor,
		cache: Optional[Dict[str, torch.Tensor]] = None,
		seq_start: int = 0,
	) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
		attn_out, new_cache = self.attn(self.norm1(x), cache=cache, seq_start=seq_start)
		x = x + attn_out
		x = x + self.mlp(self.norm2(x))
		return x, new_cache


class TransformerDynamics(nn.Module):
	"""
	Latent dynamics module that replaces the MLP transition with a causal Transformer.

	Designed for TD-MPC2:
	- Input token is [z, a, task_emb] projected to `embed_dim`.
	- Causal attention ensures step t only sees <= t history.
	- KV caching lets planning rollouts append one step at a time without recomputing history.
	- Outputs a continuous latent delta added to the previous latent (skip connection).
	"""

	def __init__(
		self,
		latent_dim: int,
		action_dim: int,
		task_dim: int,
		embed_dim: Optional[int] = None,
		num_layers: int = 2,
		num_heads: int = 4,
		mlp_ratio: float = 2.0,
		max_seq_len: int = 16,
		dropout: float = 0.0,
		attn_dropout: float = 0.0,
	):
		super().__init__()
		self.latent_dim = latent_dim
		self.action_dim = action_dim
		self.task_dim = task_dim
		self.embed_dim = embed_dim or latent_dim
		self.max_seq_len = max_seq_len

		in_dim = latent_dim + action_dim + task_dim
		self.input_proj = nn.Linear(in_dim, self.embed_dim)
		self.pos_emb = nn.Parameter(torch.zeros(1, max_seq_len, self.embed_dim))

		self.blocks = nn.ModuleList([
			TransformerBlock(
				dim=self.embed_dim,
				num_heads=num_heads,
				mlp_ratio=mlp_ratio,
				attn_dropout=attn_dropout,
				dropout=dropout,
			) for _ in range(num_layers)
		])
		self.out_norm = nn.LayerNorm(self.embed_dim)
		self.delta_head = nn.Linear(self.embed_dim, latent_dim)

		self.reset_parameters()

	def reset_parameters(self):
		nn.init.trunc_normal_(self.pos_emb, std=0.02)
		for m in self.modules():
			if isinstance(m, nn.Linear):
				nn.init.trunc_normal_(m.weight, std=0.02)
				if m.bias is not None:
					nn.init.zeros_(m.bias)
			if isinstance(m, nn.LayerNorm):
				nn.init.ones_(m.weight)
				nn.init.zeros_(m.bias)

	def init_cache(self) -> Dict[str, List[Optional[Dict[str, torch.Tensor]]]]:
		"""
		Create an empty KV cache for streaming rollout.
		"""
		return {
			"seq_len": 0,
			"kv": [None for _ in range(len(self.blocks))],
		}

	def forward(
		self,
		z: torch.Tensor,
		a: torch.Tensor,
		task_emb: Optional[torch.Tensor] = None,
		cache: Optional[Dict[str, List[Optional[Dict[str, torch.Tensor]]]]] = None,
	) -> Tuple[torch.Tensor, Dict[str, List[Optional[Dict[str, torch.Tensor]]]]]:
		"""
		Args:
			z: latent state, shape (B, latent_dim) or (B, T, latent_dim).
			a: action, same leading dims as z.
			task_emb: optional task embedding, same leading dims as z; if None, zeros are used.
			cache: optional KV cache for incremental rollout.

		Returns:
			next_z: predicted next latent(s), same leading dims as z.
			new_cache: updated cache (use for the next call when rolling out step-by-step).
		"""
		# Normalize shapes to (B, T, D)
		if z.dim() == 2:
			z = z.unsqueeze(1)
		if a.dim() == 2:
			a = a.unsqueeze(1)
		if task_emb is None:
			task_emb = z.new_zeros(z.shape[0], z.shape[1], self.task_dim)
		elif task_emb.dim() == 2:
			task_emb = task_emb.unsqueeze(1)

		B, T, _ = z.shape
		x = torch.cat([z, a, task_emb], dim=-1)
		x = self.input_proj(x)

		seq_start = cache["seq_len"] if cache is not None else 0
		pos_start = max(0, seq_start)
		pos_end = min(pos_start + T, self.max_seq_len)
		pos = self.pos_emb[:, pos_start:pos_end]
		if pos.shape[1] < T:  # wrap or repeat if sequence longer than max_seq_len
			repeats = math.ceil(T / self.max_seq_len)
			pos = self.pos_emb.repeat(1, repeats, 1)[:, :T]
		x = x + pos

		new_cache = {
			"seq_len": seq_start + T,
			"kv": [],
		}
		cache_kv = cache["kv"] if cache is not None else [None] * len(self.blocks)

		for blk, blk_cache in zip(self.blocks, cache_kv):
			x, blk_new_cache = blk(x, cache=blk_cache, seq_start=seq_start)
			new_cache["kv"].append(blk_new_cache)

		x = self.out_norm(x)
		delta = self.delta_head(x)
		next_z = z + delta  # residual update keeps latent continuous and stable

		# Remove time dimension if input was 2-D.
		if next_z.shape[1] == 1:
			next_z = next_z[:, 0]
		return next_z, new_cache
