import pathlib
import sys

import minari
import numpy as np
import torch
from tensordict import TensorDict

DATASETS = [
	"mujoco/halfcheetah/expert-v0",
	"mujoco/halfcheetah/medium-v0",
	"mujoco/halfcheetah/simple-v0",
	"mujoco/walker2d/expert-v0",
	"mujoco/walker2d/medium-v0",
	"mujoco/walker2d/simple-v0",
]


def download_all():
	"""Download all predefined Minari datasets."""
	for name in DATASETS:
		print("downloading", name)
		minari.load_dataset(name, download=True)
	print("done")


def convert_minari_to_td(dataset_name: str, out_dir: str):
	"""
	Convert a Minari dataset to the TensorDict .pt format expected by TD-MPC2 offline training.

	Assumptions:
	  - Uses state observations.
	  - Action space already scaled to [-1, 1].
	  - Aligns actions/rewards/terminated to obs_{t+1} (action at t produces obs_{t+1}).
	  - Pads each episode to a fixed length (max_episode_steps + 1) and sets terminated=1 after real data ends.
	"""
	ds = minari.load_dataset(dataset_name, download=False)
	spec_max_steps = getattr(getattr(ds.spec, "env_spec", None), "max_episode_steps", None)
	max_steps = spec_max_steps
	if max_steps is None:
		raise AttributeError(f"max_episode_steps not found in env_spec for dataset {dataset_name}")
	obs_dim = ds.observation_space.shape[0]
	act_dim = ds.action_space.shape[0]
	fixed_len = int(max_steps) + 1  # include initial observation

	obs_list, act_list, rew_list, term_list = [], [], [], []
	for ep in ds.iterate_episodes():
		obs = np.asarray(ep.observations, dtype=np.float32)            # (N+1, obs_dim)
		actions = np.asarray(ep.actions, dtype=np.float32)             # (N, act_dim)
		rewards = np.asarray(ep.rewards, dtype=np.float32)             # (N,)
		done = np.logical_or(ep.terminations, ep.truncations).astype(np.float32)  # (N,)
		N = actions.shape[0]

		obs_pad = np.zeros((fixed_len, obs_dim), dtype=np.float32)
		act_pad = np.zeros((fixed_len, act_dim), dtype=np.float32)
		rew_pad = np.zeros((fixed_len, 1), dtype=np.float32)
		term_pad = np.zeros((fixed_len, 1), dtype=np.float32)

		obs_pad[:N+1] = obs
		act_pad[1:N+1] = actions               # action_t -> obs_{t+1}
		rew_pad[1:N+1, 0] = rewards
		term_pad[1:N+1, 0] = done
		term_pad[N+1:] = 1.0                   # mark padding as terminated

		obs_list.append(torch.from_numpy(obs_pad))
		act_list.append(torch.from_numpy(act_pad))
		rew_list.append(torch.from_numpy(rew_pad))
		term_list.append(torch.from_numpy(term_pad))

	td = TensorDict({
		"obs": torch.stack(obs_list),
		"action": torch.stack(act_list),
		"reward": torch.stack(rew_list),
		"terminated": torch.stack(term_list),
	}, batch_size=[len(obs_list), fixed_len])

	out_dir = pathlib.Path(out_dir)
	out_dir.mkdir(parents=True, exist_ok=True)
	fname = dataset_name.replace("/", "-") + ".pt"
	out_path = out_dir / fname
	torch.save(td, out_path)
	print(f"saved {dataset_name} to {out_path} with shape {td.shape}")
	return out_path


if __name__ == "__main__":
	if len(sys.argv) == 1:
		download_all()
	elif len(sys.argv) == 3:
		convert_minari_to_td(sys.argv[1], sys.argv[2])
	else:
		print("Usage:\n  python data.py                 # download predefined datasets\n  python data.py <dataset_name> <out_dir>  # convert one dataset to .pt")
