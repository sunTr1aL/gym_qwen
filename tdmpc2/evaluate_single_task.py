import copy
import os
os.environ['MUJOCO_GL'] = os.getenv("MUJOCO_GL", 'egl')

import time
import hydra
import imageio
import torch
import gymnasium as gym
import numpy as np
from omegaconf import OmegaConf
from termcolor import colored

from common.parser import parse_cfg
from common.seed import set_seed
from envs import make_env
from tdmpc2 import TDMPC2


class ModelShapeWrapper(gym.Wrapper):
	"""
	Adapter that pads observations (and crops actions) to match the
	expectations of a multitask checkpoint while only running a single env.
	"""

	def __init__(self, env, obs_dim, model_action_dim):
		super().__init__(env)
		self.model_obs_dim = obs_dim
		self.model_action_dim = model_action_dim
		self.actual_action_dim = env.action_space.shape[0]
		self.observation_space = gym.spaces.Box(
			low=-np.inf, high=np.inf, shape=(self.model_obs_dim,), dtype=np.float32
		)
		self.action_space = gym.spaces.Box(
			low=-1., high=1., shape=(self.model_action_dim,), dtype=np.float32
		)
		self.max_episode_steps = env.max_episode_steps

	def _pad_obs(self, obs):
		if obs.shape[0] == self.model_obs_dim:
			return obs
		if obs.shape[0] > self.model_obs_dim:
			return obs[:self.model_obs_dim]
		pad = torch.zeros(self.model_obs_dim - obs.shape[0], dtype=obs.dtype, device=obs.device)
		return torch.cat([obs, pad], dim=0)

	def reset(self):
		return self._pad_obs(self.env.reset())

	def step(self, action):
		action = action[:self.actual_action_dim]
		obs, reward, done, info = self.env.step(action)
		return self._pad_obs(obs), reward, done, info

	def render(self, *args, **kwargs):
		return self.env.render(*args, **kwargs)


def _load_checkpoint(fp):
	ckpt = torch.load(fp, map_location='cpu', weights_only=False)
	metadata = ckpt.get('metadata', {})
	state = ckpt['model'] if 'model' in ckpt else ckpt
	return metadata, state


def _select_task_state(state, task_idx):
	filtered = {}
	for k, v in state.items():
		if k == '_task_emb.weight':
			filtered[k] = v[task_idx:task_idx+1]
		elif k == '_action_masks':
			filtered[k] = v[task_idx:task_idx+1]
		else:
			filtered[k] = v
	return filtered


@hydra.main(config_name='config', config_path='.')
def evaluate_single(cfg: OmegaConf):
	metadata, state = _load_checkpoint(cfg.checkpoint)
	task_list = metadata.get('tasks')
	if not task_list:
		raise ValueError('Checkpoint is not multitask or lacks task metadata.')
	if cfg.task not in task_list:
		raise ValueError(f'Task "{cfg.task}" not present in checkpoint tasks.')
	task_idx = task_list.index(cfg.task)

	task_dim = metadata.get('task_dim', cfg.get('task_dim', 0))
	latent_dim = metadata.get('latent_dim', cfg.get('latent_dim', 0))
	obs_dim = state['_encoder.state.0.weight'].shape[1] - task_dim
	action_dim = state['_dynamics.0.weight'].shape[1] - (latent_dim + task_dim)
	assert obs_dim > 0 and action_dim > 0

	cfg.force_multitask = True
	cfg.tasks_override = [cfg.task]
	cfg.model_size = metadata.get('model_size', cfg.get('model_size'))
	cfg.task_dim = task_dim
	cfg.obs_shape = {cfg.get('obs', 'state'): (obs_dim,)}
	cfg.action_dim = action_dim
	cfg = parse_cfg(cfg)
	if cfg.device.startswith('cuda') and not torch.cuda.is_available():
		raise RuntimeError('CUDA device requested but not available. Set device=cpu to run on CPU.')
	assert cfg.eval_episodes > 0, 'Must evaluate at least 1 episode.'
	set_seed(cfg.seed)

	env_cfg = copy.deepcopy(cfg)
	env_cfg.multitask = False
	env_cfg.force_multitask = False
	env_cfg.tasks_override = None
	env = make_env(env_cfg)
	env = ModelShapeWrapper(env, obs_dim, action_dim)
	cfg.action_dims = [env.actual_action_dim]
	cfg.episode_length = env.max_episode_steps
	cfg.episode_lengths = [cfg.episode_length]
	cfg.seed_steps = max(1000, 5*cfg.episode_length)
	cfg.record_plan_traj = False # only supported in evaluate.py

	print(colored(f'Task: {cfg.task}', 'blue', attrs=['bold']))
	print(colored(f'Model size: {cfg.get("model_size", "default")}', 'blue', attrs=['bold']))
	print(colored(f'Checkpoint: {cfg.checkpoint}', 'blue', attrs=['bold']))

	agent = TDMPC2(cfg)
	agent.load({'model': _select_task_state(state, task_idx)})

	if cfg.save_video:
		video_dir = os.path.join(cfg.work_dir, 'videos')
		os.makedirs(video_dir, exist_ok=True)

	ep_rewards, ep_successes = [], []
	for i in range(cfg.eval_episodes):
		obs, done, ep_reward, t = env.reset(), False, 0, 0
		if cfg.save_video:
			frames = [env.render()]
		while not done:
			t_start = time.perf_counter()
			action = agent.act(obs, t0=t==0, task=0)
			obs, reward, done, info = env.step(action)
			step_time = time.perf_counter() - t_start
			ep_reward += reward
			t += 1
			if cfg.get('log_step_speed', False):
				fps = (1.0 / step_time) if step_time > 0 else float('inf')
				print(colored(
					f'[{cfg.task} | ep {i+1} step {t}] {step_time*1000:.2f} ms ({fps:.1f} FPS)',
					'green'), end='\r', flush=True)
			if cfg.save_video:
				frames.append(env.render())
		if cfg.get('log_step_speed', False):
			print()
		ep_rewards.append(ep_reward)
		ep_successes.append(info['success'])
		if cfg.save_video:
			imageio.mimsave(os.path.join(video_dir, f'{cfg.task}-{i}.mp4'), frames, fps=15)

	print(colored(f'Reward: {torch.tensor(ep_rewards).float().mean().item():.01f}', 'yellow', attrs=['bold']))
	print(colored(f'Success: {torch.tensor(ep_successes).float().mean().item():.02f}', 'yellow', attrs=['bold']))


if __name__ == '__main__':
	evaluate_single()
