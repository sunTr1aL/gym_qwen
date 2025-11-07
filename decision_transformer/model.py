"""
this extremely minimal Decision Transformer model is based on
the following causal transformer (GPT) implementation:

Misha Laskin's tweet:
https://twitter.com/MishaLaskin/status/1481767788775628801?cxt=HHwWgoCzmYD9pZApAAAA

and its corresponding notebook:
https://colab.research.google.com/drive/1NUBqyboDcGte5qAJKOl8gaJC28V_73Iv?usp=sharing

** the above colab notebook has a bug while applying masked_fill 
which is fixed in the following code
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class MaskedCausalAttention(nn.Module):
    def __init__(self, h_dim, max_T, n_heads, drop_p):
        super().__init__()

        self.n_heads = n_heads
        self.max_T = max_T

        self.q_net = nn.Linear(h_dim, h_dim)
        self.k_net = nn.Linear(h_dim, h_dim)
        self.v_net = nn.Linear(h_dim, h_dim)

        self.proj_net = nn.Linear(h_dim, h_dim)

        self.att_drop = nn.Dropout(drop_p)
        self.proj_drop = nn.Dropout(drop_p)

        ones = torch.ones((max_T, max_T))
        mask = torch.tril(ones).view(1, 1, max_T, max_T)

        # register buffer makes sure mask does not get updated
        # during backpropagation
        self.register_buffer('mask',mask)

    def forward(self, x):
        B, T, C = x.shape # batch size, seq length, h_dim * n_heads

        N, D = self.n_heads, C // self.n_heads # N = num heads, D = attention dim

        # rearrange q, k, v as (B, N, T, D)
        q = self.q_net(x).view(B, T, N, D).transpose(1,2)
        k = self.k_net(x).view(B, T, N, D).transpose(1,2)
        v = self.v_net(x).view(B, T, N, D).transpose(1,2)

        # weights (B, N, T, T)
        weights = q @ k.transpose(2,3) / math.sqrt(D)
        # causal mask applied to weights
        weights = weights.masked_fill(self.mask[...,:T,:T] == 0, float('-inf'))
        # normalize weights, all -inf -> 0 after softmax
        normalized_weights = F.softmax(weights, dim=-1)

        # attention (B, N, T, D)
        attention = self.att_drop(normalized_weights @ v)

        # gather heads and project (B, N, T, D) -> (B, T, N*D)
        attention = attention.transpose(1, 2).contiguous().view(B,T,N*D)

        out = self.proj_drop(self.proj_net(attention))
        return out


class Block(nn.Module):
    def __init__(self, h_dim, max_T, n_heads, drop_p):
        super().__init__()
        self.attention = MaskedCausalAttention(h_dim, max_T, n_heads, drop_p)
        self.mlp = nn.Sequential(
                nn.Linear(h_dim, 4*h_dim),
                nn.GELU(),
                nn.Linear(4*h_dim, h_dim),
                nn.Dropout(drop_p),
            )
        self.ln1 = nn.LayerNorm(h_dim)
        self.ln2 = nn.LayerNorm(h_dim)

    def forward(self, x):
        # Attention -> LayerNorm -> MLP -> LayerNorm
        x = x + self.attention(x) # residual
        x = self.ln1(x)
        x = x + self.mlp(x) # residual
        x = self.ln2(x)
        return x


class DecisionTransformer(nn.Module):
    def __init__(self, state_dim, act_dim, n_blocks, h_dim, context_len,
                 n_heads, drop_p, max_timestep=4096):
        super().__init__()

        self.state_dim = state_dim
        self.act_dim = act_dim
        self.h_dim = h_dim

        ### transformer blocks
        input_seq_len = 3 * context_len
        blocks = [Block(h_dim, input_seq_len, n_heads, drop_p) for _ in range(n_blocks)]
        self.transformer = nn.Sequential(*blocks)

        ### projection heads (project to embedding)
        self.embed_ln = nn.LayerNorm(h_dim)
        self.embed_timestep = nn.Embedding(max_timestep, h_dim)
        self.embed_rtg = torch.nn.Linear(1, h_dim)
        self.embed_state = torch.nn.Linear(state_dim, h_dim)

        # # discrete actions
        # self.embed_action = torch.nn.Embedding(act_dim, h_dim)
        # use_action_tanh = False # False for discrete actions

        # continuous actions
        self.embed_action = torch.nn.Linear(act_dim, h_dim)
        use_action_tanh = True # True for continuous actions

        ### prediction heads
        self.predict_rtg = torch.nn.Linear(h_dim, 1)
        self.predict_state = torch.nn.Linear(h_dim, state_dim)
        self.predict_action = nn.Sequential(
            *([nn.Linear(h_dim, act_dim)] + ([nn.Tanh()] if use_action_tanh else []))
        )


    def forward(self, timesteps, states, actions, returns_to_go):

        B, T, _ = states.shape

        time_embeddings = self.embed_timestep(timesteps)

        # time embeddings are treated similar to positional embeddings
        state_embeddings = self.embed_state(states) + time_embeddings
        action_embeddings = self.embed_action(actions) + time_embeddings
        returns_embeddings = self.embed_rtg(returns_to_go) + time_embeddings

        # stack rtg, states and actions and reshape sequence as
        # (r_0, s_0, a_0, r_1, s_1, a_1, r_2, s_2, a_2 ...)
        h = torch.stack(
            (returns_embeddings, state_embeddings, action_embeddings), dim=1
        ).permute(0, 2, 1, 3).reshape(B, 3 * T, self.h_dim)

        h = self.embed_ln(h)

        # transformer and prediction
        h = self.transformer(h)

        # get h reshaped such that its size = (B x 3 x T x h_dim) and
        # h[:, 0, t] is conditioned on the input sequence r_0, s_0, a_0 ... r_t
        # h[:, 1, t] is conditioned on the input sequence r_0, s_0, a_0 ... r_t, s_t
        # h[:, 2, t] is conditioned on the input sequence r_0, s_0, a_0 ... r_t, s_t, a_t
        # that is, for each timestep (t) we have 3 output embeddings from the transformer,
        # each conditioned on all previous timesteps plus 
        # the 3 input variables at that timestep (r_t, s_t, a_t) in sequence.
        h = h.reshape(B, T, 3, self.h_dim).permute(0, 2, 1, 3)

        # get predictions
        return_preds = self.predict_rtg(h[:,2])     # predict next rtg given r, s, a
        state_preds = self.predict_state(h[:,2])    # predict next state given r, s, a
        action_preds = self.predict_action(h[:,1])  # predict action given r, s

        return state_preds, action_preds, return_preds


# ------------------------
# Qwen3 core primitives
# ------------------------

class RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x):
        dtype = x.dtype
        x = x.float()
        var = x.pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(var + self.eps)
        return (self.weight * x.to(dtype))

class RotaryEmbedding(nn.Module):
    def __init__(self, head_dim, max_position_embeddings=32768, rope_theta=1_000_000.0, device=None):
        super().__init__()
        self.head_dim = head_dim
        inv_freq = 1.0 / (rope_theta ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.max_position_embeddings = max_position_embeddings

    def forward(self, seqlen, device=None, dtype=None):
        # cos, sin shape: [1, seqlen, head_dim]
        if device is None:
            device = self.inv_freq.device
        t = torch.arange(seqlen, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)  # [seqlen, head_dim//2]
        emb = torch.cat((freqs, freqs), dim=-1)            # [seqlen, head_dim]
        cos = emb.cos().unsqueeze(0)
        sin = emb.sin().unsqueeze(0)
        if dtype is not None:
            cos = cos.to(dtype)
            sin = sin.to(dtype)
        return cos, sin

def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def apply_rope(q, k, cos, sin):
    # q, k: [B, H, T, D]    cos, sin: [1, T, D]
    q_embed = (q * cos.unsqueeze(1)) + (rotate_half(q) * sin.unsqueeze(1))
    k_embed = (k * cos.unsqueeze(1)) + (rotate_half(k) * sin.unsqueeze(1))
    return q_embed, k_embed

def repeat_kv(x, n_rep):
    # x: [B, H_kv, T, D] -> [B, H_kv*n_rep, T, D]
    if n_rep == 1:
        return x
    b, h, t, d = x.shape
    x = x[:, :, None, :, :].expand(b, h, n_rep, t, d).reshape(b, h * n_rep, t, d)
    return x

class Qwen3Attention(nn.Module):
    def __init__(
        self,
        hidden_size,
        num_heads,
        num_kv_heads,
        head_dim,
        attn_dropout=0.0,
        bias=False,
        rope=None,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.num_groups = num_heads // num_kv_heads
        self.head_dim = head_dim
        self.scale = head_dim ** -0.5
        self.q_proj = nn.Linear(hidden_size, num_heads * head_dim, bias=bias)
        self.k_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=bias)
        self.v_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=bias)
        self.o_proj = nn.Linear(num_heads * head_dim, hidden_size, bias=bias)
        self.q_norm = RMSNorm(head_dim)
        self.k_norm = RMSNorm(head_dim)
        self.attn_drop = nn.Dropout(attn_dropout)
        self.rope = rope

    def forward(self, x, attn_mask, rope=None):
        rope_ref = rope if rope is not None else self.rope
        if rope_ref is None:
            raise ValueError("Qwen3Attention requires a RotaryEmbedding instance.")
        # x: [B, T, C]
        b, t, _ = x.shape
        q = self.q_proj(x).view(b, t, self.num_heads, self.head_dim).transpose(1, 2)  # [B, H, T, D]
        k = self.k_proj(x).view(b, t, self.num_kv_heads, self.head_dim).transpose(1, 2)  # [B, H_kv, T, D]
        v = self.v_proj(x).view(b, t, self.num_kv_heads, self.head_dim).transpose(1, 2)

        # per-head RMSNorm on last dim
        q = self.q_norm(q)
        k = self.k_norm(k)

        # RoPE
        cos, sin = rope_ref(t, device=x.device, dtype=x.dtype)
        q, k = apply_rope(q, k, cos, sin)

        # GQA expand kv
        k = repeat_kv(k, self.num_groups)
        v = repeat_kv(v, self.num_groups)

        # attention
        attn_scores = torch.matmul(q, k.transpose(2, 3)) * self.scale       # [B, H, T, T]
        if attn_mask is not None:
            attn_scores = attn_scores + attn_mask
        attn_probs = F.softmax(attn_scores, dim=-1, dtype=torch.float32).to(q.dtype)
        attn_probs = self.attn_drop(attn_probs)
        ctx = torch.matmul(attn_probs, v)                                   # [B, H, T, D]
        ctx = ctx.transpose(1, 2).contiguous().view(b, t, self.num_heads * self.head_dim)
        out = self.o_proj(ctx)
        return out

class Qwen3MLP(nn.Module):
    def __init__(self, hidden_size, intermediate_size):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))

class Qwen3Block(nn.Module):
    def __init__(
        self,
        hidden_size,
        num_heads,
        num_kv_heads,
        head_dim,
        mlp_ratio=3.0,
        attn_dropout=0.0,
        bias=False,
        rope=None,
    ):
        super().__init__()
        intermediate = int(mlp_ratio * hidden_size)
        self.input_norm = RMSNorm(hidden_size)
        self.attn = Qwen3Attention(hidden_size, num_heads, num_kv_heads, head_dim, attn_dropout, bias, rope=rope)
        self.post_attn_norm = RMSNorm(hidden_size)
        self.mlp = Qwen3MLP(hidden_size, intermediate)
        self.rope = rope

    def forward(self, x, attn_mask, rope=None):
        residual = x
        x = self.input_norm(x)
        x = self.attn(x, attn_mask, rope)
        x = residual + x
        residual = x
        x = self.post_attn_norm(x)
        x = self.mlp(x)
        x = residual + x
        return x

# ------------------------
# Decision Transformer head with Qwen3 backbone
# ------------------------

class DecisionTransformerQwen3(nn.Module):
    def __init__(
        self,
        state_dim,
        act_dim,
        context_len,
        n_layers=3,
        hidden_size=128,
        num_heads=1,
        num_kv_heads=None,
        head_dim=None,
        attn_dropout=0.1,
        drop_p=0.1,
        max_timestep=4096,
        rope_theta=10_000.0,
        use_action_tanh=True
    ):
        super().__init__()

        self.state_dim = state_dim
        self.act_dim = act_dim
        self.hidden_size = hidden_size
        self.context_len = context_len

        if num_heads < 1:
            raise ValueError("num_heads must be >= 1.")
        if num_kv_heads is None:
            num_kv_heads = num_heads
        if num_kv_heads < 1 or num_heads % num_kv_heads != 0:
            raise ValueError("num_kv_heads must be >= 1 and divide num_heads.")
        if head_dim is None:
            if hidden_size % num_heads != 0:
                raise ValueError("hidden_size must be divisible by num_heads when head_dim is not provided.")
            head_dim = hidden_size // num_heads

        # embeddings for DT tokens
        self.embed_ln = RMSNorm(hidden_size)
        self.embed_timestep = nn.Embedding(max_timestep, hidden_size)
        self.embed_rtg = nn.Linear(1, hidden_size)
        self.embed_state = nn.Linear(state_dim, hidden_size)
        self.embed_action = nn.Linear(act_dim, hidden_size)

        # positional rope
        self.rope = RotaryEmbedding(head_dim=head_dim, max_position_embeddings=3 * context_len, rope_theta=rope_theta)

        # transformer blocks
        blocks = []
        for _ in range(n_layers):
            blocks.append(
                Qwen3Block(
                    hidden_size,
                    num_heads,
                    num_kv_heads,
                    head_dim,
                    mlp_ratio=3.0,
                    attn_dropout=attn_dropout,
                    bias=False,
                    rope=self.rope,
                )
            )
        self.blocks = nn.ModuleList(blocks)
        self.drop = nn.Dropout(drop_p)

        # prediction heads
        self.predict_rtg = nn.Linear(hidden_size, 1)
        self.predict_state = nn.Linear(hidden_size, state_dim)
        self.predict_action = nn.Sequential(
            nn.Linear(hidden_size, act_dim),
            *( [nn.Tanh()] if use_action_tanh else [] )
        )

        # build causal mask once at max length
        max_T = 3 * context_len
        ones = torch.ones((max_T, max_T))
        mask = torch.tril(ones).view(1, 1, max_T, max_T)
        self.register_buffer("causal_mask", mask)

    def _causal(self, T, device, dtype):
        # returns additive mask with -inf above diagonal
        m = self.causal_mask[..., :T, :T]
        return (m == 0).to(dtype=dtype) * torch.finfo(dtype).min

    def forward(self, timesteps, states, actions, returns_to_go):
        # shapes
        B, T, _ = states.shape

        time_embeddings = self.embed_timestep(timesteps)
        state_embeddings = self.embed_state(states) + time_embeddings
        action_embeddings = self.embed_action(actions) + time_embeddings
        returns_embeddings = self.embed_rtg(returns_to_go) + time_embeddings

        # interleave as r, s, a, ...
        h = torch.stack((returns_embeddings, state_embeddings, action_embeddings), dim=1) \
              .permute(0, 2, 1, 3).reshape(B, 3 * T, self.hidden_size)

        h = self.embed_ln(h)
        h = self.drop(h)

        attn_mask = self._causal(h.size(1), device=h.device, dtype=h.dtype)

        # pass through Qwen3 blocks
        for blk in self.blocks:
            h = blk(h, attn_mask)

        # reshape back to three streams
        h = h.view(B, T, 3, self.hidden_size).permute(0, 2, 1, 3)

        return_preds = self.predict_rtg(h[:, 2])     # next rtg given r, s, a
        state_preds = self.predict_state(h[:, 2])    # next state given r, s, a
        action_preds = self.predict_action(h[:, 1])  # action given r, s

        return state_preds, action_preds, return_preds
