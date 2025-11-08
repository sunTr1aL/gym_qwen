import inspect
import os
import random
import time
import pickle
from typing import Optional

import torch
import numpy as np
from torch.utils.data import Dataset
from decision_transformer.d4rl_infos import REF_MIN_SCORE, REF_MAX_SCORE, D4RL_DATASET_STATS


def discount_cumsum(x, gamma):
    disc_cumsum = np.zeros_like(x)
    disc_cumsum[-1] = x[-1]
    for t in reversed(range(x.shape[0]-1)):
        disc_cumsum[t] = x[t] + gamma * disc_cumsum[t+1]
    return disc_cumsum


def get_d4rl_normalized_score(score, env_name):
    env_key = env_name.split('-')[0].lower()
    if env_key not in REF_MAX_SCORE:
        return float('nan')
    return (score - REF_MIN_SCORE[env_key]) / (REF_MAX_SCORE[env_key] - REF_MIN_SCORE[env_key])


def get_d4rl_dataset_stats(env_d4rl_name):
    return D4RL_DATASET_STATS[env_d4rl_name]


def evaluate_on_env(model, device, context_len, env, rtg_target, rtg_scale,
                    num_eval_ep=10, max_test_ep_len=1000,
                    state_mean=None, state_std=None, render=False):

    eval_batch_size = 1  # required for forward pass

    results = {}
    total_reward = 0
    total_timesteps = 0

    state_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    if state_mean is None:
        state_mean = torch.zeros((state_dim,)).to(device)
    else:
        state_mean = torch.from_numpy(state_mean).to(device)

    if state_std is None:
        state_std = torch.ones((state_dim,)).to(device)
    else:
        state_std = torch.from_numpy(state_std).to(device)

    # same as timesteps used for training the transformer
    # also, crashes if device is passed to arange()
    timesteps = torch.arange(start=0, end=max_test_ep_len, step=1)
    timesteps = timesteps.repeat(eval_batch_size, 1).to(device)

    model.eval()

    def _model_requires_traj_mask(module) -> bool:
        try:
            sig = inspect.signature(module.forward)
        except (ValueError, TypeError):
            return False
        return any(param.name == "traj_mask" for param in sig.parameters.values())

    needs_traj_mask = _model_requires_traj_mask(model)

    with torch.no_grad():

        for _ in range(num_eval_ep):

            # zeros place holders
            actions = torch.zeros((eval_batch_size, max_test_ep_len, act_dim),
                                dtype=torch.float32, device=device)
            states = torch.zeros((eval_batch_size, max_test_ep_len, state_dim),
                                dtype=torch.float32, device=device)
            rewards_to_go = torch.zeros((eval_batch_size, max_test_ep_len, 1),
                                dtype=torch.float32, device=device)
            if needs_traj_mask:
                traj_mask = torch.zeros((eval_batch_size, max_test_ep_len),
                                        dtype=torch.float32, device=device)
            else:
                traj_mask = None

            def _forward_with_slice(start: int, end: int):
                args = (
                    timesteps[:, start:end],
                    states[:, start:end],
                    actions[:, start:end],
                    rewards_to_go[:, start:end],
                )
                if needs_traj_mask and traj_mask is not None:
                    args = args + (traj_mask[:, start:end],)
                return model.forward(*args)

            def _split_model_outputs(outputs):
                if isinstance(outputs, (tuple, list)):
                    if len(outputs) < 3:
                        raise ValueError(
                            f"Model forward must return at least 3 tensors "
                            f"(state_preds, action_preds, return_preds). Got {len(outputs)}."
                        )
                    return outputs[0], outputs[1], outputs[2]
                raise ValueError(
                    "Model forward output must be a tuple/list of tensors. "
                    f"Got type '{type(outputs).__name__}'."
                )

            # init episode
            reset_out = env.reset()
            if isinstance(reset_out, tuple):
                running_state, _ = reset_out
            else:
                running_state = reset_out
            running_reward = 0
            running_rtg = rtg_target / rtg_scale

            for t in range(max_test_ep_len):

                total_timesteps += 1

                # add state in placeholder and normalize
                state_tensor = torch.from_numpy(np.asarray(running_state, dtype=np.float32)).to(device)
                states[0, t] = (state_tensor - state_mean) / state_std

                # calcualate running rtg and add it in placeholder
                running_rtg = running_rtg - (running_reward / rtg_scale)
                rewards_to_go[0, t] = running_rtg

                if needs_traj_mask:
                    traj_mask[0, t] = 1.0

                if t < context_len:
                    model_outputs = _forward_with_slice(0, context_len)
                    _, act_preds, _ = _split_model_outputs(model_outputs)
                    act = act_preds[0, t].detach()
                else:
                    start = t - context_len + 1
                    model_outputs = _forward_with_slice(start, t + 1)
                    _, act_preds, _ = _split_model_outputs(model_outputs)
                    act = act_preds[0, -1].detach()

                step_out = env.step(act.cpu().numpy())
                if isinstance(step_out, tuple) and len(step_out) == 5:
                    running_state, running_reward, terminated, truncated, _ = step_out
                    done = bool(terminated) or bool(truncated)
                else:
                    running_state, running_reward, done, _ = step_out

                running_reward = float(running_reward)

                # add action in placeholder
                actions[0, t] = act

                total_reward += running_reward

                if render:
                    env.render()
                if done:
                    break

    results['eval/avg_reward'] = total_reward / num_eval_ep
    results['eval/avg_ep_len'] = total_timesteps / num_eval_ep

    if render and hasattr(env, "close"):
        try:
            env.close()
        except Exception:
            pass

    return results


def _load_minari_dataset(dataset_or_id):
    try:
        import minari
    except ImportError as err:
        raise ImportError(
            "minari is required to load Humanoid datasets. Install with 'pip install \"minari[hdf5]\"'."
        ) from err

    dataset = dataset_or_id
    if isinstance(dataset_or_id, str):
        dataset = minari.load_dataset(dataset_or_id)

    trajectories = []
    for episode in dataset.iterate_episodes():
        observations = episode.observations.astype(np.float32)
        actions = episode.actions.astype(np.float32)
        rewards = episode.rewards.astype(np.float32).reshape(-1)

        # Align observations/actions; Minari episodes often store final observation as extra element
        if observations.shape[0] == actions.shape[0] + 1:
            observations = observations[:-1]
        elif observations.shape[0] != actions.shape[0]:
            raise ValueError(
                f"Episode observation/action length mismatch: obs={observations.shape}, act={actions.shape}"
            )

        trajectories.append({
            'observations': observations,
            'actions': actions,
            'rewards': rewards,
        })

    return trajectories


class D4RLTrajectoryDataset(Dataset):
    def __init__(self, dataset_path, context_len, rtg_scale, minari_dataset: Optional[str] = None):

        self.context_len = context_len

        # load dataset
        if dataset_path and os.path.isfile(dataset_path):
            with open(dataset_path, 'rb') as f:
                self.trajectories = pickle.load(f)
        elif minari_dataset is not None:
            self.trajectories = _load_minari_dataset(minari_dataset)
        else:
            raise FileNotFoundError(
                f"Dataset not found at '{dataset_path}'. Provide a valid pickle file or install Minari datasets."
            )

        # calculate min len of traj, state mean and variance
        # and returns_to_go for all traj
        min_len = 10**6
        states = []
        for traj in self.trajectories:
            traj_len = traj['observations'].shape[0]
            min_len = min(min_len, traj_len)
            states.append(traj['observations'])
            # calculate returns to go and rescale them
            traj['returns_to_go'] = discount_cumsum(traj['rewards'], 1.0) / rtg_scale

        # used for input normalization
        states = np.concatenate(states, axis=0)
        self.state_mean, self.state_std = np.mean(states, axis=0), np.std(states, axis=0) + 1e-6

        # normalize states
        for traj in self.trajectories:
            traj['observations'] = (traj['observations'] - self.state_mean) / self.state_std

    def get_state_stats(self):
        return self.state_mean, self.state_std

    def __len__(self):
        return len(self.trajectories)

    def __getitem__(self, idx):
        traj = self.trajectories[idx]
        traj_len = traj['observations'].shape[0]

        if traj_len >= self.context_len:
            # sample random index to slice trajectory
            si = random.randint(0, traj_len - self.context_len)

            states = torch.from_numpy(traj['observations'][si : si + self.context_len])
            actions = torch.from_numpy(traj['actions'][si : si + self.context_len])
            returns_to_go = torch.from_numpy(traj['returns_to_go'][si : si + self.context_len])
            timesteps = torch.arange(start=si, end=si+self.context_len, step=1)

            # all ones since no padding
            traj_mask = torch.ones(self.context_len, dtype=torch.long)

        else:
            padding_len = self.context_len - traj_len

            # padding with zeros
            states = torch.from_numpy(traj['observations'])
            states = torch.cat([states,
                                torch.zeros(([padding_len] + list(states.shape[1:])),
                                dtype=states.dtype)],
                               dim=0)

            actions = torch.from_numpy(traj['actions'])
            actions = torch.cat([actions,
                                torch.zeros(([padding_len] + list(actions.shape[1:])),
                                dtype=actions.dtype)],
                               dim=0)

            returns_to_go = torch.from_numpy(traj['returns_to_go'])
            returns_to_go = torch.cat([returns_to_go,
                                torch.zeros(([padding_len] + list(returns_to_go.shape[1:])),
                                dtype=returns_to_go.dtype)],
                               dim=0)

            timesteps = torch.arange(start=0, end=self.context_len, step=1)

            traj_mask = torch.cat([torch.ones(traj_len, dtype=torch.long),
                                   torch.zeros(padding_len, dtype=torch.long)],
                                  dim=0)

        return  timesteps, states, actions, returns_to_go, traj_mask
