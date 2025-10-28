
import torch
import torch.nn as nn

from decision_transformer.models.model import TrajectoryModel
from decision_transformer.models.qwen3_transformer import QwenTransformer, QwenTransformerConfig, RMSNorm


class DecisionTransformerQwen3(TrajectoryModel):
    """
    Decision Transformer with a Qwen3-style Transformer backbone.
    Input sequence: (R_1, s_1, a_1, R_2, s_2, a_2, ...)
    We predict a_t from (R_t, s_t, a_{t-1}).
    """

    def __init__(
        self,
        state_dim,
        act_dim,
        max_length=None,
        max_ep_len=4096,
        hidden_size=2048,
        n_layer=24,
        n_head=16,
        n_kv_head=8,
        mlp_ratio=5.4,
        action_tanh=True,
        **kwargs,
    ):
        k_override = kwargs.pop("K", None)
        if k_override is not None:
            max_length = int(k_override)

        embed_dim = kwargs.pop("embed_dim", None)
        if embed_dim is not None:
            hidden_size = int(embed_dim)

        max_timestep = kwargs.pop("max_timestep", None)
        if max_timestep is not None:
            max_ep_len = int(max_timestep)

        super().__init__(state_dim, act_dim, max_length=max_length)

        self.hidden_size = hidden_size
        # Qwen-style transformer (no token embedding inside; we feed inputs_embeds)
        self.transformer = QwenTransformer(
            QwenTransformerConfig(
                d_model=hidden_size,
                n_layer=n_layer,
                n_heads=n_head,
                n_kv_heads=n_kv_head,
                mlp_ratio=mlp_ratio,
            )
        )

        self.max_ep_len = max_ep_len

        # modality embeddings
        self.embed_timestep = nn.Embedding(max_ep_len, hidden_size)
        self.embed_return = nn.Linear(1, hidden_size)
        self.embed_state = nn.Linear(state_dim, hidden_size)
        self.embed_action = nn.Linear(act_dim, hidden_size)

        # layer norm after sum of modality embeddings
        self.embed_ln = RMSNorm(hidden_size)

        # prediction heads
        self.predict_state = nn.Linear(hidden_size, self.state_dim)
        self.predict_action = nn.Sequential(
            *([nn.Linear(hidden_size, self.act_dim)] + ([nn.Tanh()] if action_tanh else []))
        )
        self.predict_return = nn.Linear(hidden_size, 1)

    def forward(self, states, actions, rewards, returns_to_go, timesteps, attention_mask=None):

        batch_size, seq_length = states.shape[0], states.shape[1]

        if attention_mask is None:
            attention_mask = torch.ones((batch_size, seq_length), dtype=torch.long, device=states.device)

        # modality embeddings
        state_embeddings = self.embed_state(states)
        action_embeddings = self.embed_action(actions)
        returns_embeddings = self.embed_return(returns_to_go)
        time_embeddings = self.embed_timestep(timesteps)

        state_embeddings = state_embeddings + time_embeddings
        action_embeddings = action_embeddings + time_embeddings
        returns_embeddings = returns_embeddings + time_embeddings

        # interleave as (R_1, s_1, a_1, R_2, s_2, a_2, ...)
        stacked_inputs = torch.stack(
            (returns_embeddings, state_embeddings, action_embeddings), dim=1
        ).permute(0, 2, 1, 3).reshape(batch_size, 3 * seq_length, self.hidden_size)

        stacked_attention_mask = torch.stack(
            (attention_mask, attention_mask, attention_mask), dim=1
        ).permute(0, 2, 1).reshape(batch_size, 3 * seq_length)

        x = self.transformer(inputs_embeds=stacked_inputs, attention_mask=stacked_attention_mask)["last_hidden_state"]

        # reshape back to (B, 3, T, H)
        x = x.reshape(batch_size, seq_length, 3, self.hidden_size).permute(0, 2, 1, 3)

        return_preds = self.predict_return(x[:, 2])  # predict next return given s,a
        state_preds = self.predict_state(x[:, 2])    # predict next state given s,a
        action_preds = self.predict_action(x[:, 1])  # predict next action given s_t

        return state_preds, action_preds, return_preds

    @torch.no_grad()
    def get_action(self, states, actions, rewards, returns_to_go, timesteps, **kwargs):
        states = states.reshape(1, -1, self.state_dim)
        actions = actions.reshape(1, -1, self.act_dim)
        returns_to_go = returns_to_go.reshape(1, -1, 1)
        timesteps = timesteps.reshape(1, -1)

        if self.max_length is not None:
            states = states[:, -self.max_length:]
            actions = actions[:, -self.max_length:]
            returns_to_go = returns_to_go[:, -self.max_length:]
            timesteps = timesteps[:, -self.max_length:]

            # left-pad to max_length
            attention_mask = torch.cat(
                [torch.zeros((1, self.max_length - states.shape[1]), device=states.device),
                 torch.ones((1, states.shape[1]), device=states.device)],
                dim=1
            ).to(dtype=torch.long)
            states = torch.cat(
                [torch.zeros((1, self.max_length - states.shape[1], self.state_dim), device=states.device), states],
                dim=1
            ).to(dtype=torch.float32)
            actions = torch.cat(
                [torch.zeros((1, self.max_length - actions.shape[1], self.act_dim), device=actions.device), actions],
                dim=1
            ).to(dtype=torch.float32)
            returns_to_go = torch.cat(
                [torch.zeros((1, self.max_length - returns_to_go.shape[1], 1), device=returns_to_go.device), returns_to_go],
                dim=1
            ).to(dtype=torch.float32)
            timesteps = torch.cat(
                [torch.zeros((1, self.max_length - timesteps.shape[1]), device=timesteps.device), timesteps],
                dim=1
            ).to(dtype=torch.long)
        else:
            attention_mask = None

        _, action_preds, _ = self.forward(
            states, actions, None, returns_to_go, timesteps, attention_mask=attention_mask, **kwargs
        )

        return action_preds[0, -1]
