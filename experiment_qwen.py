#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Experiment runner for Qwen-style Decision Transformer on Gymnasium MuJoCo tasks.

Key features in this version:
- Robust env/dataset normalization:
    * 'Humanoid-v5' + --dataset medium (new)
    * 'humanoid' + --dataset medium (new)
    * 'humanoid-medium-v2' (legacy D4RL)   -> normalized to ('Humanoid-v5', 'humanoid', 'medium')
- Minari-first dataset loading for {simple, medium, expert} via: mujoco/<env_base>/<dataset>-v0
  and automatic fallback to local pickle (e.g., humanoid-medium-v2.pkl).
- New CLI: --target_returns "6000,8000"; if omitted, auto-derive [median, p90] from dataset returns.
"""

import os
import time
import argparse
import numpy as np
import pickle
import random
import torch
import gymnasium as gym
try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = None

# ------------------------- Your project imports (adjust if names differ) -------------------------
# Make sure these modules exist in your repo; otherwise adapt import paths accordingly.
try:
    from decision_transformer.training.seq_trainer import SequenceTrainer
    from decision_transformer.evaluation.evaluate_episodes import evaluate_episode_rtg
    from decision_transformer.models.decision_transformer_qwen3 import DecisionTransformerQwen3 as QwenDecisionTransformer  # your Qwen-style DT
except Exception as e:
    raise ImportError(
        "Failed to import your project modules. Please ensure:\n"
        "  decision_transformer/training/trainer.py (SequenceTrainer)\n"
        "  decision_transformer/evaluation/evaluate_episodes.py (evaluate_episode_rtg)\n"
        "  decision_transformer/models/qwen_dt.py (QwenDecisionTransformer)\n"
        f"Original import error: {e}"
    )

# ------------------------------- Minari availability check --------------------------------------
try:
    import minari  # noqa: F401
    HAS_MINARI = True
    MINARI_ERR = None
except Exception as e:
    HAS_MINARI = False
    MINARI_ERR = e


# ================================= Helpers =================================

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _canonicalize_env_and_dataset(env_arg: str, dataset_arg: str):
    """
    Accept various spellings and normalize to:
       (gym_env_id, env_base_lower, dataset_tag)
    Examples:
      - 'Humanoid-v5' + 'medium'        -> ('Humanoid-v5', 'humanoid', 'medium')
      - 'humanoid' + 'medium'           -> ('Humanoid-v5', 'humanoid', 'medium')
      - 'humanoid-medium-v2'            -> ('Humanoid-v5', 'humanoid', 'medium')
    """
    env_map = {
        'halfcheetah': 'HalfCheetah-v5',
        'hopper':     'Hopper-v5',
        'walker2d':   'Walker2d-v5',
        'humanoid':   'Humanoid-v5',
    }
    allowed_ds = {'simple', 'medium', 'expert', 'medium-replay'}

    e = env_arg.strip()
    el = e.lower()

    # Already Gymnasium ID
    if el in {'halfcheetah-v5', 'hopper-v5', 'walker2d-v5', 'humanoid-v5'}:
        base = el.split('-')[0]
        return e, base, dataset_arg

    # D4RL-style: humanoid-medium-v2 / hopper-expert-v2 / ...
    parts = el.split('-')
    if len(parts) >= 2 and parts[0] in env_map:
        base = parts[0]
        ds = dataset_arg
        if (not ds) or (ds not in allowed_ds):
            cand = parts[1]
            if cand in allowed_ds:
                ds = cand
        return env_map[base], base, ds

    # Plain base name: 'humanoid' / 'hopper'
    if el in env_map:
        return env_map[el], el, dataset_arg

    # Fallback: keep original, best-effort base
    base = el.split('-')[0]
    return e, base, dataset_arg


def _parse_target_returns(val):
    """Parse '6000,8000' / 6000 / [6000,8000] / None -> List[float] or None."""
    if val is None:
        return None
    if isinstance(val, (list, tuple)):
        return [float(x) for x in val]
    if isinstance(val, (int, float)):
        return [float(val)]
    s = str(val).strip().replace(' ', '')
    if not s:
        return None
    return [float(x) for x in s.split(',') if x]


def _default_targets_from_returns(returns_np: np.ndarray):
    """Return [median, p90] from dataset returns as safe defaults."""
    med = float(np.percentile(returns_np, 50))
    p90 = float(np.percentile(returns_np, 90))
    if p90 <= med:
        p90 = med * 1.1
    return [med, p90]


def _resolve_pickle_path(dataset_dir: str, env_base: str, dataset: str, gym_env_id: str):
    """
    Try filenames in order:
      1) <env_base>-<dataset>-v2.pkl     e.g., humanoid-medium-v2.pkl
      2) <env_base>-<dataset>.pkl        e.g., humanoid-medium.pkl
      3) <env_base>.pkl                  e.g., humanoid.pkl
      4) <gym_env_id>.pkl                e.g., Humanoid-v5.pkl
    Search in [dataset_dir], then ./data, then ./
    """
    cands = [
        f"{env_base}-{dataset}-v2.pkl",
        f"{env_base}-{dataset}.pkl",
        f"{env_base}.pkl",
        f"{gym_env_id}.pkl",
    ]
    dirs = [dataset_dir, "data", "."]
    for d in dirs:
        for name in cands:
            p = os.path.join(d, name)
            if os.path.isfile(p):
                return p
    raise FileNotFoundError(f"Could not find offline dataset pickle. Tried names {cands} in dirs {dirs}.")


def _build_paths_from_minari_dataset(ds):
    """
    Convert Minari dataset to D4RL-style 'paths' list with keys:
      observations, next_observations, actions, rewards, terminals
    Supports typical off-by-one alignment (len(obs) == len(act)+1).
    """
    paths = []
    for ep in ds.iterate_episodes():
        obs = ep.observations.astype(np.float32)
        acts = ep.actions.astype(np.float32)
        rews = ep.rewards.astype(np.float32).reshape(-1)
        terms = ep.terminations.astype(bool).reshape(-1)

        if obs.shape[0] == acts.shape[0] + 1:
            next_obs = obs[1:].copy()
            obs = obs[:-1].copy()
        elif obs.shape[0] == acts.shape[0]:
            next_obs = np.concatenate([obs[1:], obs[-1:]], 0).copy() if obs.shape[0] > 1 else obs.copy()
        else:
            raise ValueError(f"Minari episode shapes not aligned: obs {obs.shape}, actions {acts.shape}")

        paths.append({
            "observations": obs,
            "next_observations": next_obs,
            "actions": acts,
            "rewards": rews,
            "terminals": terms,   # truncations/timeouts NOT marked True here
        })
    return paths


def _load_paths(env_base: str, dataset: str, gym_env_id: str, dataset_dir: str):
    """
    Try Minari first for {simple,medium,expert}; else fallback to local pickle.
    """
    paths = None
    if dataset in {"simple", "medium", "expert"} and HAS_MINARI:
        try:
            import h5py  # ensure hdf5 support
            from PIL import Image as _PIL_IMAGE  # ensure Pillow
            minari_id = f"mujoco/{env_base}/{dataset}-v0"
            print(f"[Minari] Loading dataset: {minari_id}")
            ds = minari.load_dataset(minari_id, download=True)
            paths = _build_paths_from_minari_dataset(ds)
        except Exception as e:
            print(f"[Minari] Failed to load: {e}. Falling back to local pickle.")
    # Fallback to local pickle
    if paths is None:
        dataset_path = _resolve_pickle_path(dataset_dir, env_base, dataset, gym_env_id)
        print(f"[PKL] Loading local dataset: {dataset_path}")
        with open(dataset_path, "rb") as f:
            paths = pickle.load(f)
    return paths


def _compute_returns(paths):
    return np.array([np.sum(p["rewards"]) for p in paths], dtype=np.float32)


def _discount_cumsum(x: np.ndarray, gamma: float = 1.0):
    """Return discounted cumulative sums of rewards."""
    out = np.zeros_like(x, dtype=np.float32)
    running = 0.0
    for t in range(len(x) - 1, -1, -1):
        running = float(x[t]) + gamma * running
        out[t] = running
    return out


def _get_state_action_sizes(paths):
    # Infer observation/action dims from first trajectory
    o = paths[0]["observations"]
    a = paths[0]["actions"]
    return o.shape[-1], a.shape[-1]


# =============================== Main experiment ===============================

def experiment(group_name: str, variant: dict):
    # --------- Normalize env/dataset
    raw_env = variant.get('env', 'Humanoid-v5')
    dataset = variant.get('dataset', 'medium')
    dataset_dir = variant.get('dataset_dir', '.')
    gym_env_id, env_base, dataset = _canonicalize_env_and_dataset(raw_env, dataset)

    # --------- Inspect env metadata (Gymnasium)
    env = gym.make(gym_env_id)
    max_episode_steps = getattr(env.spec, "max_episode_steps", 1000)
    env.close()

    # --------- Load offline trajectories (Minari first, else pkl)
    paths = _load_paths(env_base, dataset, gym_env_id, dataset_dir)

    # --------- Summaries
    returns = _compute_returns(paths)
    num_samples = int(np.sum([p["rewards"].shape[0] for p in paths]))
    print("=" * 50)
    print(f"[QWEN-DT] New experiment: {gym_env_id} {dataset}")
    print(f"{len(paths)} trajectories, {num_samples} timesteps")
    print(f"Avg return: {np.mean(returns):.2f} +- {np.std(returns):.2f}")
    print(f"Max return: {np.max(returns):.2f}, min: {np.min(returns):.2f}")
    print("=" * 50)

    # --------- Target returns (from CLI or defaults)
    trg = _parse_target_returns(variant.get('target_returns'))
    if trg is None:
        trg = _default_targets_from_returns(returns)
    variant['target_returns'] = trg
    print(f"[QWEN-DT] target_returns = {variant['target_returns']}")

    # --------- State/Action dims & normalization
    state_dim, act_dim = _get_state_action_sizes(paths)
    states = np.concatenate([p["observations"] for p in paths], axis=0)
    state_mean = states.mean(0).astype(np.float32)
    state_std = (states.std(0) + 1e-6).astype(np.float32)

    mode = variant.get('mode', 'delayed')
    pct_traj = float(variant.get('pct_traj', 1.0))
    K = int(variant['K'])
    batch_size = int(variant['batch_size'])
    num_eval_episodes = int(variant['num_eval_episodes'])
    rtg_scale = float(variant.get('scale', 1.0)) or 1.0
    device = variant['device']

    trajectories = []
    for path in paths:
        rewards = path["rewards"].astype(np.float32)
        if mode == 'delayed':
            delayed = np.zeros_like(rewards, dtype=np.float32)
            delayed[-1] = rewards.sum()
            rewards = delayed
        terminals = path.get("terminals")
        if terminals is None:
            terminals = path.get("dones")
        if terminals is None:
            terminals = np.zeros_like(rewards, dtype=np.float32)
        terminals = terminals.astype(np.float32)
        trajectories.append({
            "observations": path["observations"].astype(np.float32),
            "actions": path["actions"].astype(np.float32),
            "rewards": rewards,
            "terminals": terminals,
        })

    traj_lens = np.array([traj["actions"].shape[0] for traj in trajectories], dtype=np.int32)
    returns = np.array([traj["rewards"].sum() for traj in trajectories], dtype=np.float32)
    num_timesteps_total = int(traj_lens.sum())

    num_timesteps = max(int(pct_traj * num_timesteps_total), 1)
    sorted_inds = np.argsort(returns)
    num_trajectories = 1
    timesteps = traj_lens[sorted_inds[-1]]
    ind = len(trajectories) - 2
    while ind >= 0 and timesteps + traj_lens[sorted_inds[ind]] <= num_timesteps:
        timesteps += traj_lens[sorted_inds[ind]]
        num_trajectories += 1
        ind -= 1
    sorted_inds = sorted_inds[-num_trajectories:]
    p_sample = traj_lens[sorted_inds] / np.sum(traj_lens[sorted_inds])

    def get_batch(batch_size=batch_size, max_len=K):
        batch_inds = np.random.choice(
            np.arange(num_trajectories),
            size=batch_size,
            replace=True,
            p=p_sample,
        )

        s, a, r, d, rtg, timesteps_batch, mask = [], [], [], [], [], [], []
        for i in range(batch_size):
            traj = trajectories[int(sorted_inds[batch_inds[i]])]
            traj_len = traj["rewards"].shape[0]
            if traj_len == 0:
                continue
            si = random.randint(0, traj_len - 1)

            s_seq = traj["observations"][si:si + max_len].reshape(1, -1, state_dim)
            a_seq = traj["actions"][si:si + max_len].reshape(1, -1, act_dim)
            r_seq = traj["rewards"][si:si + max_len].reshape(1, -1, 1)
            d_seq = traj["terminals"][si:si + max_len].reshape(1, -1)

            t_seq = np.arange(si, si + s_seq.shape[1]).reshape(1, -1)
            t_seq[t_seq >= max_episode_steps] = max_episode_steps - 1

            rtg_seq = _discount_cumsum(traj["rewards"][si:], gamma=1.0)[:s_seq.shape[1] + 1].reshape(1, -1, 1)
            if rtg_seq.shape[1] <= s_seq.shape[1]:
                rtg_seq = np.concatenate([rtg_seq, np.zeros((1, 1, 1), dtype=np.float32)], axis=1)

            tlen = s_seq.shape[1]
            pad = max_len - tlen

            s_padded = np.concatenate([np.zeros((1, pad, state_dim), dtype=np.float32), s_seq], axis=1)
            s_padded = (s_padded - state_mean) / state_std
            a_padded = np.concatenate([np.ones((1, pad, act_dim), dtype=np.float32) * -10.0, a_seq], axis=1)
            r_padded = np.concatenate([np.zeros((1, pad, 1), dtype=np.float32), r_seq], axis=1)
            d_padded = np.concatenate([np.ones((1, pad), dtype=np.float32) * 2, d_seq], axis=1)
            rtg_padded = np.concatenate([np.zeros((1, pad, 1), dtype=np.float32), rtg_seq], axis=1) / rtg_scale
            t_padded = np.concatenate([np.zeros((1, pad), dtype=np.float32), t_seq], axis=1)
            m_padded = np.concatenate([np.zeros((1, pad), dtype=np.float32), np.ones((1, tlen), dtype=np.float32)], axis=1)

            s.append(s_padded)
            a.append(a_padded)
            r.append(r_padded)
            d.append(d_padded)
            rtg.append(rtg_padded)
            timesteps_batch.append(t_padded)
            mask.append(m_padded)

        s = torch.from_numpy(np.concatenate(s, axis=0)).to(dtype=torch.float32, device=device)
        a = torch.from_numpy(np.concatenate(a, axis=0)).to(dtype=torch.float32, device=device)
        r = torch.from_numpy(np.concatenate(r, axis=0)).to(dtype=torch.float32, device=device)
        d = torch.from_numpy(np.concatenate(d, axis=0)).to(dtype=torch.long, device=device)
        rtg_t = torch.from_numpy(np.concatenate(rtg, axis=0)).to(dtype=torch.float32, device=device)
        ts = torch.from_numpy(np.concatenate(timesteps_batch, axis=0)).to(dtype=torch.long, device=device)
        mask_t = torch.from_numpy(np.concatenate(mask, axis=0)).to(device=device)

        return s, a, r, d, rtg_t, ts, mask_t

    # --------- Build model (Qwen-style DT)
    model = QwenDecisionTransformer(
        state_dim=state_dim,
        act_dim=act_dim,
        K=K,
        embed_dim=int(variant['embed_dim']),
        n_layer=int(variant['n_layer']),
        n_head=int(variant['n_head']),
        n_kv_head=int(variant['n_kv_head']),
        mlp_ratio=float(variant['mlp_ratio']),
        max_timestep=max_episode_steps,
        rtg_scale=rtg_scale,
        device=device,
    ).to(device)

    # --------- Optimizer / Trainer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(variant['learning_rate']),
        weight_decay=float(variant['weight_decay'])
    )

    warmup_steps = int(variant['warmup_steps'])
    scheduler = None
    if warmup_steps > 0:
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lambda steps: min((steps + 1) / warmup_steps, 1.0),
        )

    def _make_eval_fn(target_return: float):
        def eval_fn(model):
            returns_eval, lengths_eval = [], []
            for _ in range(num_eval_episodes):
                eval_env = gym.make(gym_env_id)
                with torch.no_grad():
                    ret, length = evaluate_episode_rtg(
                        eval_env,
                        state_dim,
                        act_dim,
                        model,
                        max_ep_len=max_episode_steps,
                        scale=rtg_scale,
                        state_mean=state_mean,
                        state_std=state_std,
                        device=device,
                        target_return=target_return / rtg_scale if rtg_scale != 0 else target_return,
                        mode=mode,
                    )
                eval_env.close()
                returns_eval.append(ret)
                lengths_eval.append(length)
            return {
                f"target_{int(target_return)}_return_mean": float(np.mean(returns_eval)),
                f"target_{int(target_return)}_return_std": float(np.std(returns_eval)),
                f"target_{int(target_return)}_length_mean": float(np.mean(lengths_eval)),
                f"target_{int(target_return)}_length_std": float(np.std(lengths_eval)),
            }
        return eval_fn

    eval_fns = [_make_eval_fn(tar) for tar in variant['target_returns']]

    loss_fn = lambda _s_hat, a_hat, _r_hat, _s, a, _r: torch.mean((a_hat - a) ** 2)

    trainer = SequenceTrainer(
        model=model,
        optimizer=optimizer,
        batch_size=batch_size,
        get_batch=get_batch,
        loss_fn=loss_fn,
        scheduler=scheduler,
        eval_fns=eval_fns,
    )

    # --------- Train with progress reporting
    total_iters = int(variant['max_iters'])
    steps_per_iter = int(variant['num_steps_per_iter'])
    use_progress_bar = tqdm is not None and total_iters > 0
    iter_iterator = range(total_iters)
    if use_progress_bar:
        iter_iterator = tqdm(iter_iterator, desc="Training", unit="iter")

    for it in iter_iterator:
        iter_start = time.time()
        logs = trainer.train_iteration(num_steps=steps_per_iter)
        iter_elapsed = max(time.time() - iter_start, 1e-8)
        steps_per_sec = steps_per_iter / iter_elapsed
        iters_per_sec = 1.0 / iter_elapsed
        logs['speed/steps_per_sec'] = steps_per_sec
        logs['speed/iters_per_sec'] = iters_per_sec

        def _to_float(val):
            if isinstance(val, (int, float, np.floating)):
                return float(val)
            return None

        loss_val = _to_float(logs.get('training/train_loss_mean'))
        action_err = _to_float(logs.get('training/action_error'))
        eval_returns = {
            k.split('/')[-1]: _to_float(v)
            for k, v in logs.items()
            if k.startswith('evaluation/') and k.endswith('_return_mean')
        }

        if use_progress_bar:
            postfix = {
                'iter/s': f"{iters_per_sec:.2f}",
                'steps/s': f"{steps_per_sec:.1f}",
            }
            if loss_val is not None:
                postfix['loss'] = f"{loss_val:.4f}"
            if action_err is not None:
                postfix['act_err'] = f"{action_err:.4f}"
            for name, value in list(eval_returns.items())[:2]:
                if value is not None:
                    postfix[name] = f"{value:.1f}"
            iter_iterator.set_postfix(postfix, refresh=True)
        else:
            pieces = [
                f"iter/s: {iters_per_sec:.2f}",
                f"steps/s: {steps_per_sec:.1f}",
            ]
            if loss_val is not None:
                pieces.append(f"loss: {loss_val:.4f}")
            if action_err is not None:
                pieces.append(f"act_err: {action_err:.4f}")
            for name, value in eval_returns.items():
                if value is not None:
                    pieces.append(f"{name}: {value:.1f}")
            print(f"[iter {it}] " + " | ".join(pieces))

    if use_progress_bar:
        iter_iterator.close()

    # --------- Save checkpoint
    os.makedirs("runs", exist_ok=True)
    ckpt_path = os.path.join("runs", f"{env_base}_{dataset}_latest.pt")
    torch.save({'model': model.state_dict(), 'variant': variant}, ckpt_path)
    print(f"[QWEN-DT] Saved checkpoint to {ckpt_path}")


# =================================== CLI ===================================

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='Humanoid-v5')
    parser.add_argument('--dataset', type=str, default='medium')  # simple|medium|expert|medium-replay
    parser.add_argument('--dataset_dir', type=str, default='.',
                        help='Directory containing offline *.pkl (fallback). Default: current dir.')
    parser.add_argument('--model_type', type=str, default='qwen_dt')

    parser.add_argument('--K', type=int, default=20)
    parser.add_argument('--pct_traj', type=float, default=1.0)
    parser.add_argument('--batch_size', type=int, default=64)

    parser.add_argument('--embed_dim', type=int, default=768)
    parser.add_argument('--n_layer', type=int, default=12)
    parser.add_argument('--n_head', type=int, default=12)
    parser.add_argument('--n_kv_head', type=int, default=4)
    parser.add_argument('--mlp_ratio', type=float, default=4.0)

    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--warmup_steps', type=int, default=1000)

    parser.add_argument('--num_eval_episodes', type=int, default=10)
    parser.add_argument('--max_iters', type=int, default=5)
    parser.add_argument('--num_steps_per_iter', type=int, default=5000)

    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--scale', type=float, default=1.0, help='RTG scale')
    parser.add_argument('--mode', type=str, default='delayed')  # DT classic
    parser.add_argument('--log_to_wandb', action='store_true')
    parser.add_argument('--seed', type=int, default=42)

    # NEW: target returns
    parser.add_argument('--target_returns', type=str, default=None,
                        help="Comma-separated desired returns, e.g. '6000,8000'. "
                             "If omitted, will be auto-set from dataset (median & p90).")

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    set_seed(int(args.seed))
    experiment('gym-qwen-dt', variant=vars(args))
