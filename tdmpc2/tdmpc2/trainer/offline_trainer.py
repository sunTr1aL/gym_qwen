import os
from copy import deepcopy
from time import time
from pathlib import Path
from glob import glob

import numpy as np
import torch
from tqdm import tqdm

from tdmpc2.common.buffer import Buffer
from tdmpc2.trainer.base import Trainer


class OfflineTrainer(Trainer):
	"""Trainer class for multi-task offline TD-MPC2 training."""

	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)
		self._start_time = time()
	
	def eval(self):
		"""Evaluate a TD-MPC2 agent."""
		# Single-task evaluation mirrors the online trainer; multi-task keeps per-task eval.
		if not self.cfg.multitask:
			ep_rewards, ep_successes, ep_lengths = [], [], []
			for i in range(self.cfg.eval_episodes):
				obs, done, ep_reward, t = self.env.reset(), False, 0, 0
				if self.cfg.save_video:
					self.logger.video.init(self.env, enabled=(i==0))
				while not done:
					torch.compiler.cudagraph_mark_step_begin()
					action = self.agent.act(obs, t0=t==0, eval_mode=True)
					obs, reward, done, info = self.env.step(action)
					ep_reward += reward
					t += 1
					if self.cfg.save_video:
						self.logger.video.record(self.env)
				ep_rewards.append(ep_reward)
				ep_successes.append(info['success'])
				ep_lengths.append(t)
				if self.cfg.save_video:
					self.logger.video.save(self.agent.cfg.get("step", 0))
			return dict(
				episode_reward=np.nanmean(ep_rewards),
				episode_success=np.nanmean(ep_successes),
				episode_length=np.nanmean(ep_lengths),
			)

		results = dict()
		for task_idx in tqdm(range(len(self.cfg.tasks)), desc='Evaluating'):
			ep_rewards, ep_successes = [], []
			for _ in range(self.cfg.eval_episodes):
				obs, done, ep_reward, t = self.env.reset(task_idx), False, 0, 0
				while not done:
					torch.compiler.cudagraph_mark_step_begin()
					action = self.agent.act(obs, t0=t==0, eval_mode=True, task=task_idx)
					obs, reward, done, info = self.env.step(action)
					ep_reward += reward
					t += 1
				ep_rewards.append(ep_reward)
				ep_successes.append(info['success'])
			results.update({
				f'episode_reward+{self.cfg.tasks[task_idx]}': np.nanmean(ep_rewards),
				f'episode_success+{self.cfg.tasks[task_idx]}': np.nanmean(ep_successes),})
		return results
	
	def _load_dataset(self):
		"""Load dataset for offline training."""
		fp = Path(os.path.join(self.cfg.data_dir, '*.pt'))
		fps = sorted(glob(str(fp)))
		assert len(fps) > 0, f'No data found at {fp}'
		print(f'Found {len(fps)} files in {fp}')
		if self.cfg.multitask:
			if len(fps) < (20 if self.cfg.task == 'mt80' else 4):
				print(f'WARNING: expected 20 files for mt80 task set, 4 files for mt30 task set, found {len(fps)} files.')
		
			# Create buffer for sampling (expected sizes for mt datasets)
			_cfg = deepcopy(self.cfg)
			_cfg.episode_length = 101 if self.cfg.task == 'mt80' else 501
			_cfg.buffer_size = 550_450_000 if self.cfg.task == 'mt80' else 345_690_000
			_cfg.steps = _cfg.buffer_size
			self.buffer = Buffer(_cfg)
			for fp in tqdm(fps, desc='Loading data'):
				td = torch.load(fp, weights_only=False)
				assert td.shape[1] == _cfg.episode_length, \
					f'Expected episode length {td.shape[1]} to match config episode length {_cfg.episode_length}, ' \
					f'please double-check your config.'
				self.buffer.load(td)
			expected_episodes = _cfg.buffer_size // _cfg.episode_length
			if self.buffer.num_eps != expected_episodes:
				print(f'WARNING: buffer has {self.buffer.num_eps} episodes, expected {expected_episodes} episodes for {self.cfg.task} task set.')
			return

		# Single-task offline: infer dataset size, allocate buffer accordingly, then load.
		episode_len = None
		total_steps = 0
		for fp in fps:
			td = torch.load(fp, weights_only=False)
			ep_len = td.shape[1]
			if episode_len is None:
				episode_len = ep_len
			else:
				assert ep_len == episode_len, f'Episode length mismatch across files ({ep_len} vs {episode_len}).'
			total_steps += td.shape[0] * ep_len
		episode_len = episode_len or self.cfg.episode_length

		_cfg = deepcopy(self.cfg)
		_cfg.episode_length = episode_len
		_cfg.buffer_size = max(total_steps, self.cfg.buffer_size)
		_cfg.steps = _cfg.buffer_size
		self.buffer = Buffer(_cfg)

		for fp in tqdm(fps, desc='Loading data'):
			td = torch.load(fp, weights_only=False)
			assert td.shape[1] == _cfg.episode_length, \
				f'Expected episode length {td.shape[1]} to match config episode length {_cfg.episode_length}, please double-check your dataset.'
			self.buffer.load(td)

		expected_episodes = total_steps // _cfg.episode_length
		if self.buffer.num_eps != expected_episodes:
			print(f'WARNING: buffer has {self.buffer.num_eps} episodes, expected {expected_episodes} episodes for single-task offline set.')

	def train(self):
		"""Train a TD-MPC2 agent."""
		if self.cfg.multitask:
			assert self.cfg.task in {'mt30', 'mt80'}, \
				'Offline multitask training only supports mt30 or mt80 task sets.'
		self._load_dataset()
		
		print(f'Training agent for {self.cfg.steps} iterations...')
		metrics = {}
		pbar = tqdm(range(self.cfg.steps), desc='Training', dynamic_ncols=True, smoothing=0.1)
		for i in pbar:

			# Update agent
			train_metrics = self.agent.update(self.buffer)
			steps_per_sec = (i+1) / (time() - self._start_time)
			pbar.set_postfix({"sps": f"{steps_per_sec:.1f}"})

			# Evaluate agent periodically
			if i % self.cfg.eval_freq == 0 or i % 10_000 == 0:
				metrics = {
					'iteration': i,
					'elapsed_time': time() - self._start_time,
				}
				metrics.update(train_metrics)
				if i % self.cfg.eval_freq == 0:
					eval_metrics = self.eval()
					metrics.update(eval_metrics)
					if self.cfg.multitask:
						self.logger.pprint_multitask(metrics, self.cfg)
					if i > 0:
						self.logger.save_agent(self.agent, identifier=f'{i}')
				self.logger.log(metrics, 'pretrain')
			
		self.logger.finish(self.agent)
