"""
Distributed Data Parallel (DDP) Online Trainer for TD-MPC2.

This trainer supports multi-GPU online training where:
- Each process runs an independent environment
- Each process has its own replay buffer
- Model parameters are synchronized across GPUs using DDP
- Experience collection is parallelized across GPUs
"""

from time import time
import numpy as np
import torch
import torch.distributed as dist
from tensordict.tensordict import TensorDict
from trainer.base import Trainer


class DDPOnlineTrainer(Trainer):
	"""Trainer class for distributed online TD-MPC2 training."""

	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)
		self._step = 0
		self._ep_idx = 0
		self._start_time = time()
		self.rank = self.cfg.rank
		self.world_size = self.cfg.world_size

		# Sync frequency: how often to synchronize gradients
		self.sync_freq = getattr(self.cfg, 'sync_freq', 1)

	def common_metrics(self):
		"""Return a dictionary of current metrics."""
		elapsed_time = time() - self._start_time
		return dict(
			step=self._step,
			episode=self._ep_idx,
			elapsed_time=elapsed_time,
			steps_per_second=self._step / elapsed_time,
			rank=self.rank,
		)

	def eval(self):
		"""Evaluate a TD-MPC2 agent."""
		ep_rewards, ep_successes, ep_lengths = [], [], []
		for i in range(self.cfg.eval_episodes):
			obs, done, ep_reward, t = self.env.reset(), False, 0, 0
			if self.cfg.save_video and self.rank == 0:
				self.logger.video.init(self.env, enabled=(i==0))
			while not done:
				torch.compiler.cudagraph_mark_step_begin()
				action = self.agent.act(obs, t0=t==0, eval_mode=True)
				obs, reward, done, info = self.env.step(action)
				ep_reward += reward
				t += 1
				if self.cfg.save_video and self.rank == 0:
					self.logger.video.record(self.env)
			ep_rewards.append(ep_reward)
			ep_successes.append(info['success'])
			ep_lengths.append(t)
			if self.cfg.save_video and self.rank == 0:
				self.logger.video.save(self._step)

		# Gather metrics from all processes
		local_metrics = torch.tensor([
			np.nanmean(ep_rewards),
			np.nanmean(ep_successes),
			np.nanmean(ep_lengths),
		], device=f'cuda:{self.rank}')

		if self.world_size > 1:
			# Gather all metrics to rank 0
			gathered_metrics = [torch.zeros_like(local_metrics) for _ in range(self.world_size)]
			dist.all_gather(gathered_metrics, local_metrics)

			if self.rank == 0:
				# Average across all processes
				all_metrics = torch.stack(gathered_metrics)
				avg_metrics = all_metrics.mean(dim=0)
				return dict(
					episode_reward=avg_metrics[0].item(),
					episode_success=avg_metrics[1].item(),
					episode_length=avg_metrics[2].item(),
				)

		return dict(
			episode_reward=local_metrics[0].item(),
			episode_success=local_metrics[1].item(),
			episode_length=local_metrics[2].item(),
		)

	def to_td(self, obs, action=None, reward=None, terminated=None):
		"""Creates a TensorDict for a new episode."""
		if isinstance(obs, dict):
			obs = TensorDict(obs, batch_size=(), device='cpu')
		else:
			obs = obs.unsqueeze(0).cpu()
		if action is None:
			action = torch.full_like(self.env.rand_act(), float('nan'))
		if reward is None:
			reward = torch.tensor(float('nan'))
		if terminated is None:
			terminated = torch.tensor(float('nan'))
		td = TensorDict(
			obs=obs,
			action=action.unsqueeze(0),
			reward=reward.unsqueeze(0),
			terminated=terminated.unsqueeze(0),
			batch_size=(1,))
		return td

	def train(self):
		"""Train a TD-MPC2 agent with distributed data parallel."""
		train_metrics, done, eval_next = {}, True, False

		if self.rank == 0:
			print(f'\n{"="*50}')
			print(f'Starting DDP Online Training')
			print(f'World Size: {self.world_size}')
			print(f'Sync Frequency: {self.sync_freq}')
			print(f'Total Steps: {self.cfg.steps}')
			print(f'{"="*50}\n')

		while self._step <= self.cfg.steps:
			# Evaluate agent periodically (only on rank 0 to save time)
			if self._step % self.cfg.eval_freq == 0:
				eval_next = True

			# Reset environment
			if done:
				if eval_next:
					# Synchronize before evaluation
					if self.world_size > 1:
						dist.barrier()

					eval_metrics = self.eval()

					if self.rank == 0 and eval_metrics:
						eval_metrics.update(self.common_metrics())
						self.logger.log(eval_metrics, 'eval')

					eval_next = False

					# Synchronize after evaluation
					if self.world_size > 1:
						dist.barrier()

				if self._step > 0:
					if info['terminated'] and not self.cfg.episodic:
						if self.rank == 0:
							print('WARNING: Termination detected but episodic=false')

					train_metrics.update(
						episode_reward=torch.tensor([td['reward'] for td in self._tds[1:]]).sum(),
						episode_success=info['success'],
						episode_length=len(self._tds),
						episode_terminated=info['terminated'])
					train_metrics.update(self.common_metrics())

					# Only rank 0 logs to avoid cluttering
					if self.rank == 0:
						self.logger.log(train_metrics, 'train')

					self._ep_idx = self.buffer.add(torch.cat(self._tds))

				obs = self.env.reset()
				self._tds = [self.to_td(obs)]

			# Collect experience
			if self._step > self.cfg.seed_steps:
				action = self.agent.act(obs, t0=len(self._tds)==1)
			else:
				action = self.env.rand_act()
			obs, reward, done, info = self.env.step(action)
			self._tds.append(self.to_td(obs, action, reward, info['terminated']))

			# Update agent
			if self._step >= self.cfg.seed_steps:
				if self._step == self.cfg.seed_steps:
					num_updates = self.cfg.seed_steps
					if self.rank == 0:
						print('Pretraining agent on seed data...')
				else:
					num_updates = 1

				for update_idx in range(num_updates):
					# Control gradient synchronization
					# If sync_freq > 1, we only sync every sync_freq updates
					should_sync = ((self._step - self.cfg.seed_steps + update_idx) % self.sync_freq == 0)

					_train_metrics = self.agent.update(self.buffer, sync_gradients=should_sync)

				train_metrics.update(_train_metrics)

			self._step += 1

			# Periodic synchronization barrier
			if self.world_size > 1 and self._step % 10000 == 0:
				dist.barrier()
				if self.rank == 0:
					print(f'Step {self._step}/{self.cfg.steps} - All processes synchronized')

		# Final synchronization
		if self.world_size > 1:
			dist.barrier()

		if self.rank == 0:
			self.logger.finish(self.agent)
			print('\n' + '='*50)
			print('Training completed successfully!')
			print('='*50)
