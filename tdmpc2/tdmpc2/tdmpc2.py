import torch
import torch.nn.functional as F
from collections import deque
from typing import List, Optional, Tuple

from common import math
from common.scale import RunningScale
from common.world_model import WorldModel
from common.layers import api_model_conversion
from tensordict import TensorDict
from .speculative_manager import SpeculativeManager
from .corrector import build_corrector_from_cfg, corrector_loss
from .corrector_buffer import CorrectorBuffer


class TDMPC2(torch.nn.Module):
    """
    TD-MPC2 agent. Implements training + inference.
    Can be used for both single-task and multi-task experiments,
    and supports both state and pixel observations.
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        device_str = cfg.get('device', 'cuda')
        self.device = torch.device(device_str)
        if self.device.type != 'cuda' and getattr(self.cfg, 'compile', False):
            print('Disabling torch.compile on non-CUDA device.')
            self.cfg.compile = False
        self.model = WorldModel(cfg).to(self.device)
        self.optim = torch.optim.Adam([
            {'params': self.model._encoder.parameters(), 'lr': self.cfg.lr*self.cfg.enc_lr_scale},
            {'params': self.model._dynamics.parameters()},
            {'params': self.model._reward.parameters()},
            {'params': self.model._termination.parameters() if self.cfg.episodic else []},
            {'params': self.model._Qs.parameters()},
            {'params': self.model._task_emb.parameters() if self.cfg.multitask else []
             }
        ], lr=self.cfg.lr, capturable=True)
        self.pi_optim = torch.optim.Adam(self.model._pi.parameters(), lr=self.cfg.lr, eps=1e-5, capturable=True)
        self.model.eval()
        self.scale = RunningScale(cfg, device=self.device)
        self.cfg.iterations += 2*int(cfg.action_dim >= 20) # Heuristic for large action spaces
        self.discount = torch.tensor(
                [self._get_discount(ep_len) for ep_len in cfg.episode_lengths], device=self.device
        ) if self.cfg.multitask else torch.tensor(self._get_discount(cfg.episode_length), device=self.device)
        self._action_buffer = []
        self._spec_plan_buffer = []
        self.spec_manager = SpeculativeManager(self.model, cfg, self.device)
        self.spec_enabled = bool(getattr(self.cfg, "spec_enabled", False))
        self.spec_plan_horizon = int(getattr(self.cfg, "spec_plan_horizon", self.cfg.horizon))
        self.spec_exec_horizon = int(getattr(self.cfg, "spec_exec_horizon", self.spec_plan_horizon))
        self.spec_mismatch_threshold = float(
            getattr(self.cfg, "spec_mismatch_threshold", getattr(self.cfg, "spec_tau", 0.1))
        )
        self.spec_history_len = int(getattr(self.cfg, "spec_history_len", 4))
        self.current_plan_actions: List[torch.Tensor] = []
        self.current_plan_latents: List[torch.Tensor] = []
        self.plan_step_idx = 0
        self.steps_until_replan = 0
        self.mismatch_history: deque = deque(maxlen=self.spec_history_len)
        self._act_steps = 0
        self.corrector = None
        self.corrector_buffer = None
        if getattr(self.cfg, "use_corrector", False) or getattr(self.cfg, "collect_corrector_data", False):
            self.corrector = build_corrector_from_cfg(
                cfg,
                latent_dim=self.cfg.latent_dim,
                act_dim=self.cfg.action_dim,
                device=self.device,
            )
            ckpt_path = getattr(self.cfg, "corrector_ckpt", None)
            if ckpt_path:
                state = torch.load(ckpt_path, map_location=self.device)
                state = state.get("corrector", state)
                self.corrector.load_state_dict(state)
                print(f"Loaded corrector checkpoint from {ckpt_path}")
            buffer_capacity = getattr(self.cfg, "corrector_buffer_size", 0)
            if buffer_capacity:
                self.corrector_buffer = CorrectorBuffer(buffer_capacity, device=self.device)
        print('Episode length:', cfg.episode_length)
        print('Discount factor:', self.discount)
        prev_mean_horizon = max(self.cfg.horizon, self.spec_exec_horizon, self.spec_plan_horizon)
        self.register_buffer(
            "_prev_mean", torch.zeros(prev_mean_horizon, self.cfg.action_dim)
        )
        if cfg.compile:
            print('Compiling update function with torch.compile...')
            self._update = torch.compile(self._update, mode="reduce-overhead")

    @property
    def plan(self):
        _plan_val = getattr(self, "_plan_val", None)
        if _plan_val is not None:
            return _plan_val
        if self.cfg.compile:
            plan = torch.compile(self._plan, mode="reduce-overhead")
        else:
            plan = self._plan
        self._plan_val = plan
        return self._plan_val

    def _get_discount(self, episode_length):
        """
        Returns discount factor for a given episode length.
        Simple heuristic that scales discount linearly with episode length.
        Default values should work well for most tasks, but can be changed as needed.

        Args:
            episode_length (int): Length of the episode. Assumes episodes are of fixed length.

        Returns:
            float: Discount factor for the task.
        """
        frac = episode_length/self.cfg.discount_denom
        return min(max((frac-1)/(frac), self.cfg.discount_min), self.cfg.discount_max)

    def save(self, fp):
        """
        Save state dict of the agent to filepath.

        Args:
            fp (str): Filepath to save state dict to.
        """
        torch.save({"model": self.model.state_dict()}, fp)

    def load(self, fp):
        """
        Load a saved state dict from filepath (or dictionary) into current agent.

        Args:
            fp (str or dict): Filepath or state dict to load.
        """
        if isinstance(fp, dict):
            state_dict = fp
        else:
            state_dict = torch.load(fp, map_location=torch.get_default_device(), weights_only=False)
        state_dict = state_dict["model"] if "model" in state_dict else state_dict
        state_dict = api_model_conversion(self.model.state_dict(), state_dict)
        self.model.load_state_dict(state_dict)
        return

    @torch.no_grad()
    def act(self, obs, t0=False, eval_mode=False, task=None, return_info: bool = False):
        """Select an action by planning in the latent space of the world model."""

        self._act_steps += 1
        obs_tensor = obs.to(self.device, non_blocking=True).unsqueeze(0)
        if task is not None:
            task = torch.tensor([task], device=self.device)
        if t0:
            self._action_buffer.clear()
            self._spec_plan_buffer.clear()
            self.spec_manager.reset()
            self.current_plan_actions.clear()
            self.current_plan_latents.clear()
            self.plan_step_idx = 0
            self.steps_until_replan = 0
            self.mismatch_history.clear()
            if self.corrector_buffer is not None:
                self.corrector_buffer.reset()
        z_t = self.model.encode(obs_tensor, task)
        if self.spec_enabled:
            action, info = self._act_speculative(obs_tensor, z_t, task, eval_mode)
            if return_info:
                return action.cpu(), info
            return action.cpu()

        # Legacy speculative beam support
        chunk = int(getattr(self.cfg, "plan_chunk", 1))
        action = None
        speculative_info = None
        if self.cfg.speculate and not t0:
            speculative_info = self.spec_manager.consume(z_t, task)
            if speculative_info:
                distance = speculative_info["distance"]
                severe_miss = distance >= getattr(self.cfg, "spec_tau_miss", self.spec_manager.tau)
                self._maybe_collect_corrector_data(obs_tensor, z_t, speculative_info, task)
                candidate_action = speculative_info["action"].to(self.device)
                if (
                    getattr(self.cfg, "use_corrector", False)
                    and self.corrector is not None
                    and not severe_miss
                ):
                    candidate_action = self.corrector(
                        z_t, speculative_info["z_pred"].to(self.device), candidate_action
                    )
                if speculative_info["accepted"] and not severe_miss:
                    self._action_buffer = [a.cpu() for a in speculative_info.get("remainder", [])]
                    action = candidate_action
                elif not severe_miss:
                    self._spec_plan_buffer = [candidate_action.cpu()]
                    self._spec_plan_buffer += [a.cpu() for a in speculative_info.get("remainder", [])]
                    action = candidate_action
                else:
                    self._spec_plan_buffer = [speculative_info["action"].cpu()]
                    self._spec_plan_buffer += [a.cpu() for a in speculative_info.get("remainder", [])]

        if action is None:
            if chunk > 1:
                if self._action_buffer:
                    action = self._action_buffer.pop(0)
                elif self._spec_plan_buffer:
                    self._action_buffer = self._spec_plan_buffer
                    self._spec_plan_buffer = []
                    action = self._action_buffer.pop(0)
                else:
                    actions_seq = self.plan(obs_tensor, t0=t0, eval_mode=eval_mode, task=task, return_sequence=True).cpu()
                    actions_seq = actions_seq[:chunk]
                    self._action_buffer = [a for a in actions_seq]
                    self._spec_plan_buffer = []
                    action = self._action_buffer.pop(0)
            elif self.cfg.mpc:
                action = self.plan(obs_tensor, t0=t0, eval_mode=eval_mode, task=task)
                self._spec_plan_buffer = []
            else:
                action, info = self.model.pi(z_t, task)
                if eval_mode:
                    action = info["mean"]
                action = action[0]
                self._spec_plan_buffer = []
        selected_action = action

        if self.cfg.speculate:
            self.spec_manager.schedule(z_t, selected_action, task)
        info = None
        if speculative_info is not None:
            info = {
                "z_real": z_t.squeeze(0).detach().cpu(),
                "z_pred": speculative_info["z_pred"].detach().cpu(),
                "a_spec": speculative_info["action"].detach().cpu(),
                "distance": float(speculative_info.get("distance", 0.0)),
                "miss_flag": int(not speculative_info.get("accepted", False)),
                "accepted": bool(speculative_info.get("accepted", False)),
            }
            if return_info:
                return selected_action.cpu(), info
            return selected_action.cpu()

        return selected_action.cpu()

    def _act_speculative(self, obs_tensor: torch.Tensor, z_t: torch.Tensor, task, eval_mode: bool):
        if not self.current_plan_actions or self.steps_until_replan <= 0:
            self._compute_spec_plan(z_t, task)

        base_idx = min(self.plan_step_idx, len(self.current_plan_actions) - 1)
        z_pred_t = self.current_plan_latents[min(base_idx, len(self.current_plan_latents) - 1)]
        a_plan_t = self.current_plan_actions[base_idx]
        dist = torch.norm(z_t.squeeze(0) - z_pred_t.squeeze(0))
        miss = dist > self.spec_mismatch_threshold

        if miss:
            action = self.plan(obs_tensor, t0=False, eval_mode=eval_mode, task=task)
            self._compute_spec_plan(z_t, task)
        else:
            mismatch_hist_tensor = self._stack_mismatch_history()
            if getattr(self.cfg, "use_corrector", False) and self.corrector is not None:
                action = self.corrector(
                    z_t, z_pred_t.to(self.device), a_plan_t.to(self.device), mismatch_history=mismatch_hist_tensor
                )
            else:
                action = a_plan_t

        feature = torch.cat(
            [z_t.squeeze(0).detach().cpu(), z_pred_t.squeeze(0).detach().cpu(), (z_t - z_pred_t).squeeze(0).cpu(), a_plan_t.detach().cpu()],
            dim=-1,
        )
        self.mismatch_history.append(feature)

        self.plan_step_idx += 1
        self.steps_until_replan = max(self.steps_until_replan - 1, 0)
        if self.plan_step_idx >= len(self.current_plan_actions) and self.plan_step_idx < self.spec_exec_horizon:
            next_z = self.model.next(z_pred_t, action.unsqueeze(0), task)
            self.current_plan_latents.append(next_z)
            pi_action, pi_info = self.model.pi(next_z, task)
            if eval_mode and "mean" in pi_info:
                pi_action = pi_info["mean"]
            self.current_plan_actions.append(pi_action.squeeze(0))

        info = {
            "z_real": z_t.squeeze(0).detach().cpu(),
            "z_pred": z_pred_t.squeeze(0).detach().cpu(),
            "a_plan": a_plan_t.detach().cpu(),
            "a_spec": a_plan_t.detach().cpu(),
            "distance": float(dist.item()),
            "miss_flag": int(miss),
            "accepted": not miss,
        }
        return action, info

    def _compute_spec_plan(self, z_t: torch.Tensor, task=None) -> None:
        actions_seq = self.plan(z=z_t, t0=False, eval_mode=True, task=task, return_sequence=True, horizon=self.spec_plan_horizon)
        self.current_plan_actions = [a for a in actions_seq]
        self.current_plan_latents = [z_t.squeeze(0)]
        for act in self.current_plan_actions:
            next_z = self.model.next(self.current_plan_latents[-1].unsqueeze(0), act.unsqueeze(0), task)
            self.current_plan_latents.append(next_z.squeeze(0))
        self.plan_step_idx = 0
        self.steps_until_replan = self.spec_exec_horizon

    def _stack_mismatch_history(self) -> Optional[torch.Tensor]:
        if not self.mismatch_history:
            return None
        feats = list(self.mismatch_history)
        if len(feats) < self.spec_history_len:
            pad = [torch.zeros_like(feats[0]) for _ in range(self.spec_history_len - len(feats))]
            feats = pad + feats
        feats = feats[-self.spec_history_len :]
        return torch.stack(feats, dim=0).unsqueeze(0).to(self.device)

    def _maybe_collect_corrector_data(self, obs, z_real, speculative_info, task):
        if not getattr(self.cfg, "collect_corrector_data", False):
            return
        if self.corrector_buffer is None:
            return
        if self._act_steps % max(1, int(getattr(self.cfg, "corrector_collect_every", 1))) != 0:
            return
        if speculative_info is None:
            return
        distance = speculative_info.get("distance", 0.0)
        mismatch_eps = float(getattr(self.cfg, "corrector_mismatch_epsilon", 0.0))
        if distance <= mismatch_eps:
            return
        teacher_action = self.plan(obs, t0=False, eval_mode=True, task=task)
        self.corrector_buffer.add(
            z_real.squeeze(0).detach(),
            speculative_info["z_pred"].squeeze(0).detach(),
            speculative_info["action"].detach(),
            teacher_action.detach(),
            speculative_info.get("accepted", False),
            distance,
        )

    def save_corrector_data(self, path: str) -> None:
        if self.corrector_buffer is None:
            raise RuntimeError("Corrector buffer is not initialized.")
        self.corrector_buffer.save(path)

    @torch.no_grad()
    def plan_from_observation(self, obs, task=None, eval_mode: bool = True, horizon: Optional[int] = None):
        """Plan deterministically from a raw observation without speculation."""

        obs_tensor = torch.as_tensor(obs, device=self.device, dtype=torch.float32)
        if obs_tensor.ndim == 1:
            obs_tensor = obs_tensor.unsqueeze(0)
        return self.plan(obs_tensor, t0=False, eval_mode=eval_mode, task=task, horizon=horizon)

    @torch.no_grad()
    def plan_with_predicted_latents(
        self, obs: torch.Tensor, task=None, horizon: Optional[int] = None, eval_mode: bool = True
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """Plan a sequence and roll out predicted latents from the provided observation."""

        obs_tensor = torch.as_tensor(obs, device=self.device, dtype=torch.float32)
        if obs_tensor.ndim == 1:
            obs_tensor = obs_tensor.unsqueeze(0)
        actions_seq = self.plan(obs_tensor, t0=False, eval_mode=eval_mode, task=task, return_sequence=True, horizon=horizon)
        z = self.model.encode(obs_tensor, task)
        latents: List[torch.Tensor] = [z.squeeze(0)]
        for act in actions_seq:
            next_z = self.model.next(latents[-1].unsqueeze(0), act.unsqueeze(0), task)
            latents.append(next_z.squeeze(0))
        return actions_seq, latents

    @torch.no_grad()
    def _estimate_value(self, z, actions, task, record_traj=False):
        """
        Estimate value of a trajectory starting at latent state z and executing given actions.
        Optionally record per-step latents/rewards/Qs while rolling out.
        """
        G, discount = 0, 1
        termination = torch.zeros(self.cfg.num_samples, 1, dtype=torch.float32, device=z.device)
        traj_cache = {"z": [], "reward": [], "q": []} if record_traj else None
        dyn_cache = self.model.init_dyn_cache() if self.model.transformer_dynamic else None
        horizon = actions.shape[0]
        for t in range(horizon):
            if record_traj:
                traj_cache["z"].append(z)
            reward = math.two_hot_inv(self.model.reward(z, actions[t], task), self.cfg)
            if record_traj:
                traj_cache["reward"].append(reward)
                traj_cache["q"].append(self.model.Q(z, actions[t], task, return_type="all"))
            if self.model.transformer_dynamic:
                z, dyn_cache = self.model.next(z, actions[t], task, cache=dyn_cache, return_cache=True)
            else:
                z = self.model.next(z, actions[t], task)
            G = G + discount * (1-termination) * reward
            discount_update = self.discount[torch.tensor(task)] if self.cfg.multitask else self.discount
            discount = discount * discount_update
            if self.cfg.episodic:
                termination = torch.clip(termination + (self.model.termination(z, task) > 0.5).float(), max=1.)
        action, _ = self.model.pi(z, task)
        value = G + discount * (1-termination) * self.model.Q(z, action, task, return_type='avg')
        return (value, traj_cache) if record_traj else value

    @torch.no_grad()
    def _plan(
        self,
        obs=None,
        t0=False,
        eval_mode=False,
        task=None,
        return_sequence=False,
        horizon: Optional[int] = None,
        z: Optional[torch.Tensor] = None,
    ):
        """
        Plan a sequence of actions using the learned world model.

        Args:
            z (torch.Tensor): Latent state from which to plan.
            t0 (bool): Whether this is the first observation in the episode.
            eval_mode (bool): Whether to use the mean of the action distribution.
            task (Torch.Tensor): Task index (only used for multi-task experiments).

        Returns:
            torch.Tensor: Action to take in the environment.
        """
        horizon = horizon or self.cfg.horizon
        if z is None:
            assert obs is not None, "Either obs or z must be provided for planning."
            z = self.model.encode(obs, task)
        # Sample policy trajectories
        if self.cfg.num_pi_trajs > 0:
            pi_actions = torch.empty(horizon, self.cfg.num_pi_trajs, self.cfg.action_dim, device=self.device)
            _z = z.repeat(self.cfg.num_pi_trajs, 1)
            pi_cache = self.model.init_dyn_cache() if self.model.transformer_dynamic else None
            for t in range(horizon-1):
                pi_actions[t], _ = self.model.pi(_z, task)
                if self.model.transformer_dynamic:
                    _z, pi_cache = self.model.next(_z, pi_actions[t], task, cache=pi_cache, return_cache=True)
                else:
                    _z = self.model.next(_z, pi_actions[t], task)
            pi_actions[-1], _ = self.model.pi(_z, task)

        # Initialize state and parameters
        z = z.repeat(self.cfg.num_samples, 1)
        mean = torch.zeros(horizon, self.cfg.action_dim, device=self.device)
        std = torch.full((horizon, self.cfg.action_dim), self.cfg.max_std, dtype=torch.float, device=self.device)
        if not t0:
            mean[:-1] = self._prev_mean[1:horizon+1]
        actions = torch.empty(horizon, self.cfg.num_samples, self.cfg.action_dim, device=self.device)
        if self.cfg.num_pi_trajs > 0:
            actions[:, :self.cfg.num_pi_trajs] = pi_actions

        # Iterate MPPI
        record_traj = getattr(self.cfg, "record_plan_traj", False)
        traj_cache = None
        for _ in range(self.cfg.iterations):

            # Sample actions
            r = torch.randn(horizon, self.cfg.num_samples-self.cfg.num_pi_trajs, self.cfg.action_dim, device=std.device)
            actions_sample = mean.unsqueeze(1) + std.unsqueeze(1) * r
            actions_sample = actions_sample.clamp(-1, 1)
            actions[:, self.cfg.num_pi_trajs:] = actions_sample
            if self.cfg.multitask:
                actions = actions * self.model._action_masks[task]

            # Compute elite actions
            if record_traj:
                value, traj_cache = self._estimate_value(z, actions, task, record_traj=True)
                value = value.nan_to_num(0)
            else:
                value = self._estimate_value(z, actions, task).nan_to_num(0)
            elite_idxs = torch.topk(value.squeeze(1), self.cfg.num_elites, dim=0).indices
            elite_value, elite_actions = value[elite_idxs], actions[:, elite_idxs]

            # Update parameters
            max_value = elite_value.max(0).values
            score = torch.exp(self.cfg.temperature*(elite_value - max_value))
            score = score / score.sum(0)
            mean = (score.unsqueeze(0) * elite_actions).sum(dim=1) / (score.sum(0) + 1e-9)
            std = ((score.unsqueeze(0) * (elite_actions - mean.unsqueeze(1)) ** 2).sum(dim=1) / (score.sum(0) + 1e-9)).sqrt()
            std = std.clamp(self.cfg.min_std, self.cfg.max_std)
            if self.cfg.multitask:
                mean = mean * self.model._action_masks[task]
                std = std * self.model._action_masks[task]

        # Select action
        rand_idx = math.gumbel_softmax_sample(score.squeeze(1))
        actions = torch.index_select(elite_actions, 1, rand_idx).squeeze(1)

        # Optional: record the chosen trajectory for debugging/logging without extra model calls.
        if record_traj and traj_cache is not None:
            chosen_idx = elite_idxs[rand_idx]
            traj = {"actions": actions.detach().cpu(), "z": [], "reward": [], "q": []}
            for h in range(horizon):
                traj["z"].append(traj_cache["z"][h][chosen_idx].detach().cpu())
                traj["reward"].append(traj_cache["reward"][h][chosen_idx].detach().cpu())
                traj["q"].append(traj_cache["q"][h][:, chosen_idx].detach().cpu())
            traj["z"] = torch.stack(traj["z"])
            traj["reward"] = torch.stack(traj["reward"])
            traj["q"] = torch.stack(traj["q"])
            self.last_plan_traj = traj

        a, std = actions[0], std[0]
        if not eval_mode:
            a = a + std * torch.randn(self.cfg.action_dim, device=std.device)
        self._prev_mean[:horizon].copy_(mean)
        if return_sequence:
            return actions.clamp(-1, 1)
        return a.clamp(-1, 1)

    def update_pi(self, zs, task):
        """
        Update policy using a sequence of latent states.

        Args:
            zs (torch.Tensor): Sequence of latent states.
            task (torch.Tensor): Task index (only used for multi-task experiments).

        Returns:
            float: Loss of the policy update.
        """
        action, info = self.model.pi(zs, task)
        qs = self.model.Q(zs, action, task, return_type='avg', detach=True)
        self.scale.update(qs[0])
        qs = self.scale(qs)

        # Loss is a weighted sum of Q-values
        rho = torch.pow(self.cfg.rho, torch.arange(len(qs), device=self.device))
        pi_loss = (-(self.cfg.entropy_coef * info["scaled_entropy"] + qs).mean(dim=(1,2)) * rho).mean()
        pi_loss.backward()
        pi_grad_norm = torch.nn.utils.clip_grad_norm_(self.model._pi.parameters(), self.cfg.grad_clip_norm)
        self.pi_optim.step()
        self.pi_optim.zero_grad(set_to_none=True)

        info = TensorDict({
            "pi_loss": pi_loss,
            "pi_grad_norm": pi_grad_norm,
            "pi_entropy": info["entropy"],
            "pi_scaled_entropy": info["scaled_entropy"],
            "pi_scale": self.scale.value,
        })
        return info

    @torch.no_grad()
    def _td_target(self, next_z, reward, terminated, task):
        """
        Compute the TD-target from a reward and the observation at the following time step.

        Args:
            next_z (torch.Tensor): Latent state at the following time step.
            reward (torch.Tensor): Reward at the current time step.
            terminated (torch.Tensor): Termination signal at the current time step.
            task (torch.Tensor): Task index (only used for multi-task experiments).

        Returns:
            torch.Tensor: TD-target.
        """
        action, _ = self.model.pi(next_z, task)
        discount = self.discount[task].unsqueeze(-1) if self.cfg.multitask else self.discount
        return reward + discount * (1-terminated) * self.model.Q(next_z, action, task, return_type='min', target=True)

    def _update(self, obs, action, reward, terminated, task=None):
        # Compute targets
        with torch.no_grad():
            next_z = self.model.encode(obs[1:], task)
            td_targets = self._td_target(next_z, reward, terminated, task)

        # Prepare for update
        self.model.train()

        # Latent rollout
        zs = torch.empty(self.cfg.horizon+1, self.cfg.batch_size, self.cfg.latent_dim, device=self.device)
        z = self.model.encode(obs[0], task)
        zs[0] = z
        consistency_loss = 0
        dyn_cache = self.model.init_dyn_cache() if self.model.transformer_dynamic else None
        for t, (_action, _next_z) in enumerate(zip(action.unbind(0), next_z.unbind(0))):
            if self.model.transformer_dynamic:
                z, dyn_cache = self.model.next(z, _action, task, cache=dyn_cache, return_cache=True)
            else:
                z = self.model.next(z, _action, task)
            consistency_loss = consistency_loss + F.mse_loss(z, _next_z) * self.cfg.rho**t
            zs[t+1] = z

        # Predictions
        _zs = zs[:-1]
        qs = self.model.Q(_zs, action, task, return_type='all')
        reward_preds = self.model.reward(_zs, action, task)
        if self.cfg.episodic:
            termination_pred = self.model.termination(zs[1:], task, unnormalized=True)

        # Compute losses
        reward_loss, value_loss = 0, 0
        for t, (rew_pred_unbind, rew_unbind, td_targets_unbind, qs_unbind) in enumerate(zip(reward_preds.unbind(0), reward.unbind(0), td_targets.unbind(0), qs.unbind(1))):
            reward_loss = reward_loss + math.soft_ce(rew_pred_unbind, rew_unbind, self.cfg).mean() * self.cfg.rho**t
            for _, qs_unbind_unbind in enumerate(qs_unbind.unbind(0)):
                value_loss = value_loss + math.soft_ce(qs_unbind_unbind, td_targets_unbind, self.cfg).mean() * self.cfg.rho**t

        consistency_loss = consistency_loss / self.cfg.horizon
        reward_loss = reward_loss / self.cfg.horizon
        if self.cfg.episodic:
            termination_loss = F.binary_cross_entropy_with_logits(termination_pred, terminated)
        else:
            termination_loss = 0.
        value_loss = value_loss / (self.cfg.horizon * self.cfg.num_q)
        total_loss = (
            self.cfg.consistency_coef * consistency_loss +
            self.cfg.reward_coef * reward_loss +
            self.cfg.termination_coef * termination_loss +
            self.cfg.value_coef * value_loss
        )

        # Update model
        total_loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip_norm)
        self.optim.step()
        self.optim.zero_grad(set_to_none=True)

        # Update policy
        pi_info = self.update_pi(zs.detach(), task)

        # Update target Q-functions
        self.model.soft_update_target_Q()

        # Return training statistics
        self.model.eval()
        info = TensorDict({
            "consistency_loss": consistency_loss,
            "reward_loss": reward_loss,
            "value_loss": value_loss,
            "termination_loss": termination_loss,
            "total_loss": total_loss,
            "grad_norm": grad_norm,
        })
        if self.cfg.episodic:
            info.update(math.termination_statistics(torch.sigmoid(termination_pred[-1]), terminated[-1]))
        info.update(pi_info)
        return info.detach().mean()

    def update(self, buffer):
        """
        Main update function. Corresponds to one iteration of model learning.

        Args:
            buffer (common.buffer.Buffer): Replay buffer.

        Returns:
            dict: Dictionary of training statistics.
        """
        obs, action, reward, terminated, task = buffer.sample()
        if getattr(self.cfg, "debug_buffer_shapes", False):
            print(f"[update] obs {obs.shape}, action {action.shape}, reward {reward.shape}, terminated {terminated.shape if terminated is not None else None}, task {task.shape if task is not None else None}")
        kwargs = {}
        if task is not None:
            kwargs["task"] = task
        torch.compiler.cudagraph_mark_step_begin()
        return self._update(obs, action, reward, terminated, **kwargs)
