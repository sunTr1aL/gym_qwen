from copy import deepcopy

import torch
import torch.nn as nn

from tdmpc2.common import layers, math, init
from tdmpc2.tdmpc_transformer_dynamic import TransformerDynamics
from tensordict import TensorDict


class WorldModel(nn.Module):
    """
    TD-MPC2 implicit world model architecture.
    Can be used for both single-task and multi-task experiments.
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.transformer_dynamic = bool(getattr(cfg, "transformer_dynamic", False))
        task_dim = int(getattr(cfg, "task_dim", 0))
        if hasattr(cfg, "task_emb_dim"):
            task_dim = int(getattr(cfg, "task_emb_dim", task_dim))
            cfg.task_dim = task_dim
        if cfg.multitask:
            self._task_emb = nn.Embedding(len(cfg.tasks), task_dim, max_norm=1)

            tasks = getattr(cfg, "tasks", None)
            num_tasks = len(tasks) if isinstance(tasks, (list, tuple)) else 1
            try:
                action_dim_scalar = int(cfg.action_dim)
            except Exception as exc:
                raise AssertionError(
                    f"cfg.action_dim must be numeric for multitask models (got {cfg.action_dim!r})"
                ) from exc

            raw_action_dims = getattr(cfg, "action_dims", None)
            action_dims = None
            if isinstance(raw_action_dims, (list, tuple)):
                action_dims = [int(d) for d in raw_action_dims]
            elif raw_action_dims is not None and not isinstance(raw_action_dims, str):
                action_dims = [int(raw_action_dims)] * num_tasks
            else:
                # Fallback to cfg.action_dim when action_dims is missing or non-numeric
                action_dims = [action_dim_scalar] * num_tasks

            if len(action_dims) != num_tasks:
                action_dims = (action_dims * num_tasks)[:num_tasks]

            self.register_buffer("_action_masks", torch.zeros(num_tasks, action_dim_scalar))
            for i in range(num_tasks):
                self._action_masks[i, :action_dims[i]] = 1.
        self._encoder = layers.enc(cfg)
        if self.transformer_dynamic:
            self._dynamics = TransformerDynamics(
                latent_dim=cfg.latent_dim,
                action_dim=cfg.action_dim,
                task_dim=task_dim,
                embed_dim=getattr(cfg, "transformer_embed_dim", None),
                num_layers=getattr(cfg, "transformer_layers", 2),
                num_heads=getattr(cfg, "transformer_heads", 4),
                mlp_ratio=getattr(cfg, "transformer_mlp_ratio", 2.0),
                max_seq_len=getattr(cfg, "transformer_max_seq_len", cfg.horizon + 4),
                dropout=getattr(cfg, "dropout", 0.0),
                attn_dropout=getattr(cfg, "transformer_attn_dropout", 0.0),
            )
        else:
            dyn_in_dim = getattr(cfg, "dyn_in_dim", None)
            if dyn_in_dim is None:
                dyn_in_dim = cfg.latent_dim + cfg.action_dim + task_dim
            self._dynamics = layers.mlp(dyn_in_dim, 2*[cfg.mlp_dim], cfg.latent_dim, act=layers.SimNorm(cfg))
        rew_in_dim = getattr(cfg, "rew_in_dim", None)
        if rew_in_dim is None:
            rew_in_dim = cfg.latent_dim + cfg.action_dim + task_dim
        self._reward = layers.mlp(rew_in_dim, 2*[cfg.mlp_dim], max(cfg.num_bins, 1))
        self._termination = layers.mlp(cfg.latent_dim + task_dim, 2*[cfg.mlp_dim], 1) if cfg.episodic else None
        pi_in_dim = getattr(cfg, "pi_in_dim", None)
        if pi_in_dim is None:
            pi_in_dim = cfg.latent_dim + task_dim
        self._pi = layers.mlp(pi_in_dim, 2*[cfg.mlp_dim], 2*cfg.action_dim)
        q_in_dim = getattr(cfg, "dyn_in_dim", None)
        if q_in_dim is None:
            q_in_dim = cfg.latent_dim + cfg.action_dim + task_dim
        self._Qs = layers.Ensemble([layers.mlp(q_in_dim, 2*[cfg.mlp_dim], max(cfg.num_bins, 1), dropout=cfg.dropout) for _ in range(cfg.num_q)])
        self.apply(init.weight_init)
        q_param_keys = []
        if hasattr(self._Qs, "params") and hasattr(self._Qs.params, "keys"):
            key_variants = (
                {"args": (True,), "kwargs": {}},
                {"args": (), "kwargs": {"include_nested": True}},
                {"args": (), "kwargs": {}},
            )
            for variant in key_variants:
                try:
                    q_param_keys = list(
                        self._Qs.params.keys(*variant["args"], **variant["kwargs"])
                    )
                    break
                except TypeError:
                    continue
        if not q_param_keys and isinstance(self._Qs.params, dict):
            q_param_keys = list(self._Qs.params.keys())

        q_weight_keys = [
            key
            for key in q_param_keys
            if isinstance(key, tuple)
            and key[-1] == "weight"
            and (len(key) < 2 or key[-2] != "ln")
        ]

        def _layer_order(key):
            try:
                return int(key[0])
            except (TypeError, ValueError):
                return -1

        q_weight_key = max(q_weight_keys, key=_layer_order) if q_weight_keys else None

        params_to_zero = [self._reward[-1].weight]
        if q_weight_key is not None:
            params_to_zero.append(self._Qs.params[q_weight_key])
        init.zero_(params_to_zero)

        self.register_buffer("log_std_min", torch.tensor(cfg.log_std_min))
        self.register_buffer("log_std_dif", torch.tensor(cfg.log_std_max) - self.log_std_min)
        self.init()

    def init(self):
        # Create detached and target Q copies without registering extra ParamModules
        self._detach_Qs_params = None
        self._target_Qs_params = None
        self._detach_Qs = deepcopy(self._Qs)
        self._target_Qs = deepcopy(self._Qs)
        for p in self._detach_Qs.parameters():
            p.requires_grad_(False)
        for p in self._target_Qs.parameters():
            p.requires_grad_(False)

    def __repr__(self):
        repr = 'TD-MPC2 World Model\n'
        modules = ['Encoder', 'Dynamics', 'Reward', 'Termination', 'Policy prior', 'Q-functions']
        for i, m in enumerate([self._encoder, self._dynamics, self._reward, self._termination, self._pi, self._Qs]):
            if m == self._termination and not self.cfg.episodic:
                continue
            repr += f"{modules[i]}: {m}\n"
        repr += "Learnable parameters: {:,}".format(self.total_params)
        return repr

    @property
    def total_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        self.init()
        return self

    def train(self, mode=True):
        """
        Overriding `train` method to keep target Q-networks in eval mode.
        """
        super().train(mode)
        self._target_Qs.train(False)
        return self

    def init_dyn_cache(self):
        """
        Initialize cache for transformer dynamics (no-op for MLP dynamics).
        """
        if not self.transformer_dynamic:
            return None
        return self._dynamics.init_cache()

    def soft_update_target_Q(self):
        """
        Soft-update target Q-networks using Polyak averaging.
        """
        with torch.no_grad():
            for target_param, online_param in zip(
                self._target_Qs.parameters(), self._detach_Qs.parameters()
            ):
                target_param.lerp_(online_param, self.cfg.tau)

    def task_emb(self, x, task):
        """
        Continuous task embedding for multi-task experiments.
        Retrieves the task embedding for a given task ID `task`
        and concatenates it to the input `x`.
        """
        if isinstance(task, int):
            task = torch.tensor([task], device=x.device)
        emb = self._task_emb(task.long())
        if x.ndim == 3:
            emb = emb.unsqueeze(0).repeat(x.shape[0], 1, 1)
        elif emb.shape[0] == 1:
            emb = emb.repeat(x.shape[0], 1)
        return torch.cat([x, emb], dim=-1)

    def encode(self, obs, task):
        """
        Encodes an observation into its latent representation.
        This implementation assumes a single state-based observation.
        """
        if self.cfg.multitask:
            obs = self.task_emb(obs, task)
        if self.cfg.obs == 'rgb' and obs.ndim == 5:
            return torch.stack([self._encoder[self.cfg.obs](o) for o in obs])
        return self._encoder[self.cfg.obs](obs)

    def next(self, z, a, task, cache=None, return_cache=False):
        """
        Predicts the next latent state given the current latent state and action.
        """
        if self.transformer_dynamic:
            task_emb = None
            if self.cfg.multitask:
                task_tensor = task if isinstance(task, torch.Tensor) else torch.tensor([task], device=z.device)
                if task_tensor.ndim == 0:
                    task_tensor = task_tensor.unsqueeze(0)
                task_emb = self._task_emb(task_tensor.long())
                if z.ndim == 3: # (B, T, latent) or (T, B, latent)
                    if z.shape[0] == task_emb.shape[0]:
                        task_emb = task_emb.unsqueeze(1).expand(-1, z.shape[1], -1)
                    else:
                        task_emb = task_emb.unsqueeze(0).expand(z.shape[0], -1, -1)
                action_mask = self._action_masks[task_tensor]
                if a.ndim == 3 and action_mask.ndim == 2:
                    action_mask = action_mask.unsqueeze(0).expand(a.shape[0], -1, -1)
                a = a * action_mask
            next_z, cache = self._dynamics(z, a, task_emb, cache=cache)
            return (next_z, cache) if return_cache else next_z

        if self.cfg.multitask:
            z = self.task_emb(z, task)
        z = torch.cat([z, a], dim=-1)
        out = self._dynamics(z)
        return (out, cache) if return_cache else out

    def reward(self, z, a, task):
        """
        Predicts instantaneous (single-step) reward.
        """
        if self.cfg.multitask:
            z = self.task_emb(z, task)
        z = torch.cat([z, a], dim=-1)
        return self._reward(z)
    
    def termination(self, z, task, unnormalized=False):
        """
        Predicts termination signal.
        """
        assert task is None
        if self.cfg.multitask:
            z = self.task_emb(z, task)
        if unnormalized:
            return self._termination(z)
        return torch.sigmoid(self._termination(z))
        

    def pi(self, z, task):
        """
        Samples an action from the policy prior.
        The policy prior is a Gaussian distribution with
        mean and (log) std predicted by a neural network.
        """
        if self.cfg.multitask:
            z = self.task_emb(z, task)

        # Gaussian policy prior
        mean, log_std = self._pi(z).chunk(2, dim=-1)
        log_std = math.log_std(log_std, self.log_std_min, self.log_std_dif)
        eps = torch.randn_like(mean)

        if self.cfg.multitask: # Mask out unused action dimensions
            mean = mean * self._action_masks[task]
            log_std = log_std * self._action_masks[task]
            eps = eps * self._action_masks[task]
            action_dims = self._action_masks.sum(-1)[task].unsqueeze(-1)
        else: # No masking
            action_dims = None

        log_prob = math.gaussian_logprob(eps, log_std)

        # Scale log probability by action dimensions
        size = eps.shape[-1] if action_dims is None else action_dims
        scaled_log_prob = log_prob * size

        # Reparameterization trick
        action = mean + eps * log_std.exp()
        mean, action, log_prob = math.squash(mean, action, log_prob)

        entropy_scale = scaled_log_prob / (log_prob + 1e-8)
        info = TensorDict({
            "mean": mean,
            "log_std": log_std,
            "action_prob": 1.,
            "entropy": -log_prob,
            "scaled_entropy": -log_prob * entropy_scale,
        })
        return action, info

    def Q(self, z, a, task, return_type='min', target=False, detach=False):
        """
        Predict state-action value.
        `return_type` can be one of [`min`, `avg`, `all`]:
            - `min`: return the minimum of two randomly subsampled Q-values.
            - `avg`: return the average of two randomly subsampled Q-values.
            - `all`: return all Q-values.
        `target` specifies whether to use the target Q-networks or not.
        """
        assert return_type in {'min', 'avg', 'all'}

        if self.cfg.multitask:
            z = self.task_emb(z, task)

        z = torch.cat([z, a], dim=-1)
        if target:
            qnet = self._target_Qs
        elif detach:
            qnet = self._detach_Qs
        else:
            qnet = self._Qs
        out = qnet(z)

        if return_type == 'all':
            return out

        qidx = torch.randperm(self.cfg.num_q, device=out.device)[:2]
        Q = math.two_hot_inv(out[qidx], self.cfg)
        if return_type == "min":
            return Q.min(0).values
        return Q.sum(0) / 2
