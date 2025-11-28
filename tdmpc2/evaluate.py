import os
os.environ['MUJOCO_GL'] = os.getenv("MUJOCO_GL", 'egl')
import warnings
warnings.filterwarnings('ignore')

import time
import hydra
import imageio
import numpy as np
import torch
from termcolor import colored

from common.parser import parse_cfg
from common.seed import set_seed
from envs import make_env
from tdmpc2 import TDMPC2

torch.backends.cudnn.benchmark = True


@hydra.main(config_name='config', config_path='.')
def evaluate(cfg: dict):
	"""
	Script for evaluating a single-task / multi-task TD-MPC2 checkpoint.

	Most relevant args:
		`task`: task name (or mt30/mt80 for multi-task evaluation)
		`model_size`: model size, must be one of `[1, 5, 19, 48, 317]` (default: 5)
		`checkpoint`: path to model checkpoint to load
		`eval_episodes`: number of episodes to evaluate on per task (default: 10)
		`save_video`: whether to save a video of the evaluation (default: True)
		`seed`: random seed (default: 1)
	
	See config.yaml for a full list of args.

	Example usage:
	````
		$ python evaluate.py task=mt80 model_size=48 checkpoint=/path/to/mt80-48M.pt
		$ python evaluate.py task=mt30 model_size=317 checkpoint=/path/to/mt30-317M.pt
		$ python evaluate.py task=dog-run checkpoint=/path/to/dog-1.pt save_video=true
	```
	"""
	cfg = parse_cfg(cfg)
	if cfg.device.startswith('cuda') and not torch.cuda.is_available():
		raise RuntimeError('CUDA device requested but not available. Set device=cpu to run on CPU.')
	assert cfg.eval_episodes > 0, 'Must evaluate at least 1 episode.'
	set_seed(cfg.seed)
	print(colored(f'Task: {cfg.task}', 'blue', attrs=['bold']))
	print(colored(f'Model size: {cfg.get("model_size", "default")}', 'blue', attrs=['bold']))
	print(colored(f'Checkpoint: {cfg.checkpoint}', 'blue', attrs=['bold']))
	if not cfg.multitask and ('mt80' in cfg.checkpoint or 'mt30' in cfg.checkpoint):
		print(colored('Warning: single-task evaluation of multi-task models is not currently supported.', 'red', attrs=['bold']))
		print(colored('To evaluate a multi-task model, use task=mt80 or task=mt30.', 'red', attrs=['bold']))

	# Make environment
	env = make_env(cfg)

	# Load agent
	agent = TDMPC2(cfg)
	assert os.path.exists(cfg.checkpoint), f'Checkpoint {cfg.checkpoint} not found! Must be a valid filepath.'
	agent.load(cfg.checkpoint)
	
	# Evaluate
	if cfg.multitask:
		print(colored(f'Evaluating agent on {len(cfg.tasks)} tasks:', 'yellow', attrs=['bold']))
	else:
		print(colored(f'Evaluating agent on {cfg.task}:', 'yellow', attrs=['bold']))
	if cfg.save_video:
		video_dir = os.path.join(cfg.work_dir, 'videos')
		os.makedirs(video_dir, exist_ok=True)
	if cfg.get('record_plan_traj', False):
		traj_dir = os.path.join(cfg.work_dir, cfg.get('plan_traj_subdir', 'plan_traj'))
		os.makedirs(traj_dir, exist_ok=True)
	scores = []
	tasks = cfg.tasks if cfg.multitask else [cfg.task]
	for task_idx, task in enumerate(tasks):
		if not cfg.multitask:
			task_idx = None
		ep_rewards, ep_successes = [], []
		for i in range(cfg.eval_episodes):
			step_trajs = [] if cfg.get('record_plan_traj', False) else None
			obs, done, ep_reward, t = env.reset(task_idx=task_idx), False, 0, 0
			if cfg.save_video:
				frames = [env.render()]
			while not done:
				t_start = time.perf_counter()
				action = agent.act(obs, t0=t==0, task=task_idx)
				if step_trajs is not None and getattr(agent, "last_plan_traj", None) is not None:
					step_trajs.append({
						"step": t,
						"task": task,
						"traj": agent.last_plan_traj
					})
				obs, reward, done, info = env.step(action)
				step_time = time.perf_counter() - t_start
				ep_reward += reward
				t += 1
				if cfg.get('log_step_speed', False):
					fps = (1.0 / step_time) if step_time > 0 else float('inf')
					print(colored(
						f'[{task} | ep {i+1} step {t}] {step_time*1000:.2f} ms ({fps:.1f} FPS)',
						'green'), end='\r', flush=True)
				if cfg.save_video:
					frames.append(env.render())
			if cfg.get('log_step_speed', False):
				print() # newline after episode logging
			ep_rewards.append(ep_reward)
			ep_successes.append(info['success'])
			if cfg.save_video:
				imageio.mimsave(
					os.path.join(video_dir, f'{task}-{i}.mp4'), frames, fps=15)
			if step_trajs is not None:
				save_fp = os.path.join(traj_dir, f'{task}-{i}.pt')
				torch.save({
					"task": task,
					"episode": i,
					"step_trajs": step_trajs,
				}, save_fp)
		ep_rewards = np.mean(ep_rewards)
		ep_successes = np.mean(ep_successes)
		if cfg.multitask:
			scores.append(ep_successes*100 if task.startswith('mw-') else ep_rewards/10)
		print(colored(f'  {task:<22}' \
			f'\tR: {ep_rewards:.01f}  ' \
			f'\tS: {ep_successes:.02f}', 'yellow'))
	if cfg.multitask:
		print(colored(f'Normalized score: {np.mean(scores):.02f}', 'yellow', attrs=['bold']))


if __name__ == '__main__':
	evaluate()
