"""
Distributed Data Parallel (DDP) wrapper for TD-MPC2 agent.

This module provides a DDP-wrapped version of the TD-MPC2 agent
for multi-GPU training. The model components are wrapped with
PyTorch's DistributedDataParallel for automatic gradient synchronization.
"""

import torch
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP

from tdmpc2 import TDMPC2
from common import math
from common.layers import api_model_conversion
from common.scale import RunningScale
from common.world_model import WorldModel
from tensordict import TensorDict


class TDMPC2_DDP(TDMPC2):
	"""
	DDP-wrapped TD-MPC2 agent for distributed training.

	This class wraps the world model with DistributedDataParallel
	to enable multi-GPU training with automatic gradient synchronization.
	"""

	def __init__(self, cfg):
		"""
		Initialize DDP-wrapped TD-MPC2 agent.

		Args:
			cfg: Configuration object with rank and world_size attributes
		"""
		# Don't call parent __init__ directly, we'll customize it
		torch.nn.Module.__init__(self)

		self.cfg = cfg
		self.rank = cfg.rank
		self.world_size = cfg.world_size

		# Set device based on rank
		device_str = f'cuda:{cfg.rank}'
		self.device = torch.device(device_str)

		# Disable torch.compile in DDP mode (can cause issues)
		if getattr(self.cfg, 'compile', False):
			print(f'[Rank {self.rank}] Disabling torch.compile in DDP mode for stability')
			self.cfg.compile = False

		# Create world model
		self.model = WorldModel(cfg).to(self.device)

		# Wrap model with DDP
		# find_unused_parameters=False for better performance
		# gradient_as_bucket_view=True for memory efficiency
		self.model = DDP(
			self.model,
			device_ids=[cfg.rank],
			output_device=cfg.rank,
			find_unused_parameters=False,
			gradient_as_bucket_view=True,
			broadcast_buffers=True,
		)

		# Access the underlying module for creating optimizers
		model_module = self.model.module

		# Create optimizers
		self.optim = torch.optim.Adam([
			{'params': model_module._encoder.parameters(), 'lr': self.cfg.lr*self.cfg.enc_lr_scale},
			{'params': model_module._dynamics.parameters()},
			{'params': model_module._reward.parameters()},
			{'params': model_module._termination.parameters() if self.cfg.episodic else []},
			{'params': model_module._Qs.parameters()},
			{'params': model_module._task_emb.parameters() if self.cfg.multitask else []}
		], lr=self.cfg.lr, capturable=False)  # capturable=False for DDP

		self.pi_optim = torch.optim.Adam(
			model_module._pi.parameters(),
			lr=self.cfg.lr,
			eps=1e-5,
			capturable=False  # capturable=False for DDP
		)

		self.model.eval()

		# Running scale for Q-value normalization
		self.scale = RunningScale(cfg, device=self.device)

		# Adjust iterations for large action spaces
		self.cfg.iterations += 2*int(cfg.action_dim >= 20)

		# Discount factor
		self.discount = torch.tensor(
			[self._get_discount(ep_len) for ep_len in cfg.episode_lengths], device=self.device
		) if self.cfg.multitask else torch.tensor(self._get_discount(cfg.episode_length), device=self.device)

		# Action buffer for chunked planning
		self._action_buffer = []

		if self.rank == 0:
			print('Episode length:', cfg.episode_length)
			print('Discount factor:', self.discount)

		# Previous mean for planning
		self._prev_mean = torch.nn.Buffer(
			torch.zeros(self.cfg.horizon, self.cfg.action_dim, device=self.device)
		)

		if self.rank == 0:
			print(f'\nDDP Model initialized on {self.world_size} GPUs')
			print(f'Rank {self.rank} device: {self.device}')

	@property
	def plan(self):
		"""Property to get the planning function."""
		_plan_val = getattr(self, "_plan_val", None)
		if _plan_val is not None:
			return _plan_val
		# No compile in DDP mode
		self._plan_val = self._plan
		return self._plan_val

	def save(self, fp):
		"""
		Save state dict of the agent to filepath.
		Only saves the underlying module (without DDP wrapper).

		Args:
			fp (str): Filepath to save state dict to.
		"""
		# Save only from rank 0 to avoid conflicts
		if self.rank == 0:
			torch.save({"model": self.model.module.state_dict()}, fp)

	def load(self, fp):
		"""
		Load a saved state dict from filepath (or dictionary) into current agent.

		Args:
			fp (str or dict): Filepath or state dict to load.
		"""
		if isinstance(fp, dict):
			state_dict = fp
		else:
			state_dict = torch.load(fp, map_location=self.device, weights_only=False)

		state_dict = state_dict["model"] if "model" in state_dict else state_dict
		state_dict = api_model_conversion(self.model.module.state_dict(), state_dict)

		# Load into the underlying module
		self.model.module.load_state_dict(state_dict)
		return

	@torch.no_grad()
	def act(self, obs, t0=False, eval_mode=False, task=None):
		"""
		Select an action by planning in the latent space of the world model.

		Args:
			obs (torch.Tensor): Observation from the environment.
			t0 (bool): Whether this is the first observation in the episode.
			eval_mode (bool): Whether to use the mean of the action distribution.
			task (int): Task index (only used for multi-task experiments).

		Returns:
			torch.Tensor: Action to take in the environment.
		"""
		obs = obs.to(self.device, non_blocking=True).unsqueeze(0)
		if task is not None:
			task = torch.tensor([task], device=self.device)

		chunk = int(getattr(self.cfg, "plan_chunk", 1))
		if t0:
			self._action_buffer.clear()

		if chunk > 1:
			if self._action_buffer:
				return self._action_buffer.pop(0)
			actions_seq = self.plan(obs, t0=t0, eval_mode=eval_mode, task=task, return_sequence=True).cpu()
			actions_seq = actions_seq[:chunk]
			self._action_buffer = [a for a in actions_seq]
			return self._action_buffer.pop(0)

		if self.cfg.mpc:
			return self.plan(obs, t0=t0, eval_mode=eval_mode, task=task).cpu()

		# Use model.module to access the underlying model
		z = self.model.module.encode(obs, task)
		action, info = self.model.module.pi(z, task)
		if eval_mode:
			action = info["mean"]
		return action[0].cpu()

	def update(self, buffer, sync_gradients=True):
		"""
		Main update function. Corresponds to one iteration of model learning.

		Args:
			buffer (common.buffer.Buffer): Replay buffer.
			sync_gradients (bool): Whether to synchronize gradients in this update.
				If False, gradients are accumulated locally without synchronization.

		Returns:
			dict: Dictionary of training statistics.
		"""
		obs, action, reward, terminated, task = buffer.sample()

		if getattr(self.cfg, "debug_buffer_shapes", False) and self.rank == 0:
			print(f"[update] obs {obs.shape}, action {action.shape}, reward {reward.shape}, "
				  f"terminated {terminated.shape if terminated is not None else None}, "
				  f"task {task.shape if task is not None else None}")

		kwargs = {}
		if task is not None:
			kwargs["task"] = task

		# Control gradient synchronization via DDP context manager
		if sync_gradients:
			# Normal mode: synchronize gradients
			torch.compiler.cudagraph_mark_step_begin()
			return self._update(obs, action, reward, terminated, **kwargs)
		else:
			# No-sync mode: accumulate gradients locally without synchronization
			# This can be used for gradient accumulation
			with self.model.no_sync():
				torch.compiler.cudagraph_mark_step_begin()
				return self._update(obs, action, reward, terminated, **kwargs)

	def _update(self, obs, action, reward, terminated, task=None):
		"""
		Internal update function. Same as parent class but works with DDP-wrapped model.

		Args:
			obs: Observations
			action: Actions
			reward: Rewards
			terminated: Termination flags
			task: Task indices (for multi-task)

		Returns:
			TensorDict with training metrics
		"""
		# Access the underlying module for all forward passes
		model = self.model.module

		# Compute targets
		with torch.no_grad():
			next_z = model.encode(obs[1:], task)
			td_targets = self._td_target(next_z, reward, terminated, task)

		# Prepare for update
		self.model.train()

		# Latent rollout
		zs = torch.empty(self.cfg.horizon+1, self.cfg.batch_size, self.cfg.latent_dim, device=self.device)
		z = model.encode(obs[0], task)
		zs[0] = z
		consistency_loss = 0
		dyn_cache = model.init_dyn_cache() if model.transformer_dynamic else None

		for t, (_action, _next_z) in enumerate(zip(action.unbind(0), next_z.unbind(0))):
			if model.transformer_dynamic:
				z, dyn_cache = model.next(z, _action, task, cache=dyn_cache, return_cache=True)
			else:
				z = model.next(z, _action, task)
			consistency_loss = consistency_loss + F.mse_loss(z, _next_z) * self.cfg.rho**t
			zs[t+1] = z

		# Predictions
		_zs = zs[:-1]
		qs = model.Q(_zs, action, task, return_type='all')
		reward_preds = model.reward(_zs, action, task)
		if self.cfg.episodic:
			termination_pred = model.termination(zs[1:], task, unnormalized=True)

		# Compute losses
		reward_loss, value_loss = 0, 0
		for t, (rew_pred_unbind, rew_unbind, td_targets_unbind, qs_unbind) in enumerate(
			zip(reward_preds.unbind(0), reward.unbind(0), td_targets.unbind(0), qs.unbind(1))
		):
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

		# Update model (gradients are synchronized by DDP)
		total_loss.backward()
		grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip_norm)
		self.optim.step()
		self.optim.zero_grad(set_to_none=True)

		# Update policy
		pi_info = self.update_pi(zs.detach(), task)

		# Update target Q-functions
		model.soft_update_target_Q()

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
		model = self.model.module
		action, _ = model.pi(next_z, task)
		discount = self.discount[task].unsqueeze(-1) if self.cfg.multitask else self.discount
		return reward + discount * (1-terminated) * model.Q(next_z, action, task, return_type='min', target=True)

	def update_pi(self, zs, task):
		"""
		Update policy using a sequence of latent states.

		Args:
			zs (torch.Tensor): Sequence of latent states.
			task (torch.Tensor): Task index (only used for multi-task experiments).

		Returns:
			TensorDict: Dictionary with policy training metrics.
		"""
		model = self.model.module

		action, info = model.pi(zs, task)
		qs = model.Q(zs, action, task, return_type='avg', detach=True)
		self.scale.update(qs[0])
		qs = self.scale(qs)

		# Loss is a weighted sum of Q-values
		rho = torch.pow(self.cfg.rho, torch.arange(len(qs), device=self.device))
		pi_loss = (-(self.cfg.entropy_coef * info["scaled_entropy"] + qs).mean(dim=(1,2)) * rho).mean()
		pi_loss.backward()
		pi_grad_norm = torch.nn.utils.clip_grad_norm_(model._pi.parameters(), self.cfg.grad_clip_norm)
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
	def _estimate_value(self, z, actions, task, record_traj=False):
		"""
		Estimate value of a trajectory starting at latent state z and executing given actions.
		"""
		model = self.model.module

		G, discount = 0, 1
		termination = torch.zeros(self.cfg.num_samples, 1, dtype=torch.float32, device=z.device)
		traj_cache = {"z": [], "reward": [], "q": []} if record_traj else None
		dyn_cache = model.init_dyn_cache() if model.transformer_dynamic else None

		for t in range(self.cfg.horizon):
			if record_traj:
				traj_cache["z"].append(z)
			reward = math.two_hot_inv(model.reward(z, actions[t], task), self.cfg)
			if record_traj:
				traj_cache["reward"].append(reward)
				traj_cache["q"].append(model.Q(z, actions[t], task, return_type="all"))
			if model.transformer_dynamic:
				z, dyn_cache = model.next(z, actions[t], task, cache=dyn_cache, return_cache=True)
			else:
				z = model.next(z, actions[t], task)
			G = G + discount * (1-termination) * reward
			discount_update = self.discount[torch.tensor(task)] if self.cfg.multitask else self.discount
			discount = discount * discount_update
			if self.cfg.episodic:
				termination = torch.clip(termination + (model.termination(z, task) > 0.5).float(), max=1.)

		action, _ = model.pi(z, task)
		value = G + discount * (1-termination) * model.Q(z, action, task, return_type='avg')
		return (value, traj_cache) if record_traj else value

	@torch.no_grad()
	def _plan(self, obs, t0=False, eval_mode=False, task=None, return_sequence=False):
		"""
		Plan a sequence of actions using the learned world model.
		Same as parent class but uses model.module.
		"""
		model = self.model.module

		# Sample policy trajectories
		z = model.encode(obs, task)
		if self.cfg.num_pi_trajs > 0:
			pi_actions = torch.empty(self.cfg.horizon, self.cfg.num_pi_trajs, self.cfg.action_dim, device=self.device)
			_z = z.repeat(self.cfg.num_pi_trajs, 1)
			pi_cache = model.init_dyn_cache() if model.transformer_dynamic else None
			for t in range(self.cfg.horizon-1):
				pi_actions[t], _ = model.pi(_z, task)
				if model.transformer_dynamic:
					_z, pi_cache = model.next(_z, pi_actions[t], task, cache=pi_cache, return_cache=True)
				else:
					_z = model.next(_z, pi_actions[t], task)
			pi_actions[-1], _ = model.pi(_z, task)

		# Initialize state and parameters
		z = z.repeat(self.cfg.num_samples, 1)
		mean = torch.zeros(self.cfg.horizon, self.cfg.action_dim, device=self.device)
		std = torch.full((self.cfg.horizon, self.cfg.action_dim), self.cfg.max_std, dtype=torch.float, device=self.device)
		if not t0:
			mean[:-1] = self._prev_mean[1:]
		actions = torch.empty(self.cfg.horizon, self.cfg.num_samples, self.cfg.action_dim, device=self.device)
		if self.cfg.num_pi_trajs > 0:
			actions[:, :self.cfg.num_pi_trajs] = pi_actions

		# Iterate MPPI
		record_traj = getattr(self.cfg, "record_plan_traj", False)
		traj_cache = None
		for _ in range(self.cfg.iterations):
			# Sample actions
			r = torch.randn(self.cfg.horizon, self.cfg.num_samples-self.cfg.num_pi_trajs, self.cfg.action_dim, device=std.device)
			actions_sample = mean.unsqueeze(1) + std.unsqueeze(1) * r
			actions_sample = actions_sample.clamp(-1, 1)
			actions[:, self.cfg.num_pi_trajs:] = actions_sample
			if self.cfg.multitask:
				actions = actions * model._action_masks[task]

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
				mean = mean * model._action_masks[task]
				std = std * model._action_masks[task]

		# Select action
		rand_idx = math.gumbel_softmax_sample(score.squeeze(1))
		actions = torch.index_select(elite_actions, 1, rand_idx).squeeze(1)

		# Optional: record trajectory
		if record_traj and traj_cache is not None:
			chosen_idx = elite_idxs[rand_idx]
			traj = {"actions": actions.detach().cpu(), "z": [], "reward": [], "q": []}
			for h in range(self.cfg.horizon):
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
		self._prev_mean.copy_(mean)
		if return_sequence:
			return actions.clamp(-1, 1)
		return a.clamp(-1, 1)
