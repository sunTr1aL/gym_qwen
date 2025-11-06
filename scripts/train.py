import argparse
import os
import sys
import random
import csv
import math
from datetime import datetime

import numpy as np
import gymnasium as gym

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from decision_transformer.utils import D4RLTrajectoryDataset, evaluate_on_env, get_d4rl_normalized_score
from decision_transformer.model import DecisionTransformer, DecisionTransformerQwen3


def _create_env(env_name):
    try:
        return gym.make(env_name)
    except Exception as gymnasium_err:
        raise RuntimeError(
            f"Failed to construct environment '{env_name}' with gymnasium ({gymnasium_err}). "
            "Ensure required environment packages (e.g., mujoco, gymnasium-robotics) are installed."
        )


def _resolve_qwen_dims(args):
    hidden_size = max(1, args.embed_dim)
    requested_heads = max(1, args.n_heads)

    resolved_heads = requested_heads
    if hidden_size >= 128 and resolved_heads < 2:
        resolved_heads = 2

    if hidden_size % resolved_heads != 0:
        gcd_heads = math.gcd(hidden_size, resolved_heads)
        resolved_heads = max(1, gcd_heads)

    head_dim = args.head_dim if args.head_dim is not None else max(1, hidden_size // resolved_heads)
    hidden_size = head_dim * resolved_heads

    num_kv_heads = args.num_kv_heads if args.num_kv_heads is not None else max(1, resolved_heads // 2 or 1)
    if resolved_heads % num_kv_heads != 0:
        num_kv_heads = max(1, math.gcd(resolved_heads, num_kv_heads))

    return hidden_size, resolved_heads, num_kv_heads, head_dim


def _build_model(args, state_dim, act_dim, device):
    model_choice = args.model.lower()
    if model_choice == 'dt':
        model = DecisionTransformer(
            state_dim=state_dim,
            act_dim=act_dim,
            n_blocks=args.n_blocks,
            h_dim=args.embed_dim,
            context_len=args.context_len,
            n_heads=args.n_heads,
            drop_p=args.dropout_p,
        )
        info = f"hidden={args.embed_dim}, heads={args.n_heads}"
    elif model_choice == 'qwen3':
        hidden_size, num_heads, num_kv_heads, head_dim = _resolve_qwen_dims(args)
        adjustments = []
        if hidden_size != args.embed_dim:
            adjustments.append(f"hidden {args.embed_dim}->{hidden_size}")
        if num_heads != args.n_heads:
            adjustments.append(f"heads {args.n_heads}->{num_heads}")
        if args.num_kv_heads is not None and num_kv_heads != args.num_kv_heads:
            adjustments.append(f"kv_heads {args.num_kv_heads}->{num_kv_heads}")
        if args.head_dim is not None and head_dim != args.head_dim:
            adjustments.append(f"head_dim {args.head_dim}->{head_dim}")
        model = DecisionTransformerQwen3(
            state_dim=state_dim,
            act_dim=act_dim,
            context_len=args.context_len,
            n_layers=args.n_blocks,
            hidden_size=hidden_size,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            attn_dropout=args.attn_dropout,
            drop_p=args.dropout_p,
            max_timestep=args.max_timestep,
            rope_theta=args.rope_theta,
        )
        info = f"hidden={hidden_size}, heads={num_heads}, kv_heads={num_kv_heads}, head_dim={head_dim}"
        if adjustments:
            info += " | " + ", ".join(adjustments)
    else:
        raise ValueError(f"Unsupported model '{args.model}'. Expected 'dt' or 'qwen3'.")

    return model.to(device), info


def train(args):

    dataset = args.dataset          # medium / medium-replay / medium-expert
    rtg_scale = args.rtg_scale      # normalize returns to go

    # use v3 env for evaluation because
    # Decision Transformer paper evaluates results on v3 envs

    minari_dataset = None
    rtg_target = None

    if args.env == 'walker2d':
        env_name = 'Walker2d-v3'
        rtg_target = 5000
        env_d4rl_name = f'walker2d-{dataset}-v2'

    elif args.env == 'halfcheetah':
        env_name = 'HalfCheetah-v3'
        rtg_target = 6000
        env_d4rl_name = f'halfcheetah-{dataset}-v2'

    elif args.env == 'hopper':
        env_name = 'Hopper-v3'
        rtg_target = 3600
        env_d4rl_name = f'hopper-{dataset}-v2'

    elif args.env == 'humanoid':
        env_name = 'Humanoid-v5'
        env_d4rl_name = f'humanoid-{dataset}-v5'
        minari_dataset = f'mujoco/humanoid/{dataset}-v0'

    else:
        raise NotImplementedError

    max_eval_ep_len = args.max_eval_ep_len  # max len of one episode
    num_eval_ep = args.num_eval_ep          # num of evaluation episodes

    batch_size = args.batch_size            # training batch size
    lr = args.lr                            # learning rate
    wt_decay = args.wt_decay                # weight decay
    warmup_steps = args.warmup_steps        # warmup steps for lr scheduler

    # total updates = max_train_iters x num_updates_per_iter
    max_train_iters = args.max_train_iters
    num_updates_per_iter = args.num_updates_per_iter

    context_len = args.context_len      # K in decision transformer

    # load data from this file
    dataset_path = f'{args.dataset_dir}/{env_d4rl_name}.pkl'

    # saves model and csv in this directory
    log_dir = args.log_dir
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # training and evaluation device
    device = torch.device(args.device)

    start_time = datetime.now().replace(microsecond=0)
    start_time_str = start_time.strftime("%y-%m-%d-%H-%M-%S")

    model_choice = args.model.lower()
    if model_choice == 'dt':
        model_tag = 'dt'
    elif model_choice == 'qwen3':
        model_tag = 'dtqwen3'
    else:
        raise ValueError(f"Unsupported model '{args.model}'. Expected 'dt' or 'qwen3'.")
    prefix = f"{model_tag}_{env_d4rl_name}"

    save_model_name =  prefix + "_model_" + start_time_str + ".pt"
    save_model_path = os.path.join(log_dir, save_model_name)
    save_best_model_path = save_model_path[:-3] + "_best.pt"

    log_csv_name = prefix + "_log_" + start_time_str + ".csv"
    log_csv_path = os.path.join(log_dir, log_csv_name)

    csv_writer = csv.writer(open(log_csv_path, 'a', 1))
    csv_header = (["duration", "num_updates", "action_loss",
                   "eval_avg_reward", "eval_avg_ep_len", "eval_d4rl_score"])

    csv_writer.writerow(csv_header)

    print("=" * 60)
    print("start time: " + start_time_str)
    print("=" * 60)

    dataset_source = dataset_path
    if not os.path.isfile(dataset_path) and minari_dataset is not None:
        dataset_source = f"minari:{minari_dataset}"

    print("device set to: " + str(device))
    print("dataset source: " + dataset_source)
    print("model architecture: " + model_choice)
    print("model save path: " + save_model_path)
    print("log csv save path: " + log_csv_path)

    traj_dataset = D4RLTrajectoryDataset(
                        dataset_path,
                        context_len,
                        rtg_scale,
                        minari_dataset=minari_dataset
                    )

    traj_data_loader = DataLoader(
                            traj_dataset,
                            batch_size=batch_size,
                            shuffle=True,
                            pin_memory=True,
                            drop_last=True
                        )

    data_iter = iter(traj_data_loader)

    ## get state stats from dataset
    state_mean, state_std = traj_dataset.get_state_stats()

    # derive RTG target from dataset statistics if not predefined
    if rtg_target is None:
        dataset_returns = np.array([np.sum(traj['rewards']) for traj in traj_dataset.trajectories])
        if dataset_returns.size == 0:
            raise ValueError('Dataset appears to be empty; cannot derive RTG target.')
        rtg_target = float(np.percentile(dataset_returns, 90))
        print(f"derived rtg target (p90 of dataset returns): {rtg_target:.2f}")

    env = _create_env(env_name)

    state_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    model, model_info = _build_model(args, state_dim, act_dim, device)
    print("model resolved dims: " + model_info)
    param_total = sum(p.numel() for p in model.parameters())
    print(f"model parameter count: {param_total / 1e6:.3f}M")

    optimizer = torch.optim.AdamW(
                        model.parameters(),
                        lr=lr,
                        weight_decay=wt_decay
                    )

    scheduler = torch.optim.lr_scheduler.LambdaLR(
                            optimizer,
                            lambda steps: min((steps+1)/warmup_steps, 1)
                        )

    max_d4rl_score = -1.0
    total_updates = 0

    for i_train_iter in range(max_train_iters):

        log_action_losses = []
        model.train()

        for _ in range(num_updates_per_iter):
            try:
                timesteps, states, actions, returns_to_go, traj_mask = next(data_iter)
            except StopIteration:
                data_iter = iter(traj_data_loader)
                timesteps, states, actions, returns_to_go, traj_mask = next(data_iter)

            timesteps = timesteps.to(device)    # B x T
            states = states.to(device)          # B x T x state_dim
            actions = actions.to(device)        # B x T x act_dim
            returns_to_go = returns_to_go.to(device).unsqueeze(dim=-1) # B x T x 1
            traj_mask = traj_mask.to(device)    # B x T
            action_target = torch.clone(actions).detach().to(device)

            state_preds, action_preds, return_preds = model.forward(
                                                            timesteps=timesteps,
                                                            states=states,
                                                            actions=actions,
                                                            returns_to_go=returns_to_go
                                                        )
            # only consider non padded elements
            action_preds = action_preds.view(-1, act_dim)[traj_mask.view(-1,) > 0]
            action_target = action_target.view(-1, act_dim)[traj_mask.view(-1,) > 0]

            action_loss = F.mse_loss(action_preds, action_target, reduction='mean')

            optimizer.zero_grad()
            action_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.25)
            optimizer.step()
            scheduler.step()

            log_action_losses.append(action_loss.detach().cpu().item())

        # evaluate action accuracy
        results = evaluate_on_env(model, device, context_len, env, rtg_target, rtg_scale,
                                num_eval_ep, max_eval_ep_len, state_mean, state_std)

        eval_avg_reward = results['eval/avg_reward']
        eval_avg_ep_len = results['eval/avg_ep_len']
        eval_d4rl_score = get_d4rl_normalized_score(results['eval/avg_reward'], env_name) * 100

        mean_action_loss = np.mean(log_action_losses)
        time_elapsed = str(datetime.now().replace(microsecond=0) - start_time)

        total_updates += num_updates_per_iter

        log_str = ("=" * 60 + '\n' +
                "time elapsed: " + time_elapsed  + '\n' +
                "num of updates: " + str(total_updates) + '\n' +
                "action loss: " +  format(mean_action_loss, ".5f") + '\n' +
                "eval avg reward: " + format(eval_avg_reward, ".5f") + '\n' +
                "eval avg ep len: " + format(eval_avg_ep_len, ".5f") + '\n' +
                "eval d4rl score: " + format(eval_d4rl_score, ".5f")
            )

        print(log_str)

        log_data = [time_elapsed, total_updates, mean_action_loss,
                    eval_avg_reward, eval_avg_ep_len,
                    eval_d4rl_score]

        csv_writer.writerow(log_data)

        # save model
        print("max d4rl score: " + format(max_d4rl_score, ".5f"))
        if eval_d4rl_score >= max_d4rl_score:
            print("saving max d4rl score model at: " + save_best_model_path)
            torch.save(model.state_dict(), save_best_model_path)
            max_d4rl_score = eval_d4rl_score

        print("saving current model at: " + save_model_path)
        torch.save(model.state_dict(), save_model_path)


    print("=" * 60)
    print("finished training!")
    print("=" * 60)
    end_time = datetime.now().replace(microsecond=0)
    time_elapsed = str(end_time - start_time)
    end_time_str = end_time.strftime("%y-%m-%d-%H-%M-%S")
    print("started training at: " + start_time_str)
    print("finished training at: " + end_time_str)
    print("total training time: " + time_elapsed)
    print("max d4rl score: " + format(max_d4rl_score, ".5f"))
    print("saved max d4rl score model at: " + save_best_model_path)
    print("saved last updated model at: " + save_model_path)
    print("=" * 60)



if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('--env', type=str, default='halfcheetah')
    parser.add_argument('--dataset', type=str, default='medium')
    parser.add_argument('--rtg_scale', type=int, default=1000)

    parser.add_argument('--max_eval_ep_len', type=int, default=1000)
    parser.add_argument('--num_eval_ep', type=int, default=10)

    parser.add_argument('--dataset_dir', type=str, default='data/')
    parser.add_argument('--log_dir', type=str, default='dt_runs/')

    parser.add_argument('--model', type=str, default='dt', choices=['dt', 'qwen3'])
    parser.add_argument('--context_len', type=int, default=20)
    parser.add_argument('--n_blocks', type=int, default=3)
    parser.add_argument('--embed_dim', type=int, default=128)
    parser.add_argument('--n_heads', type=int, default=1)
    parser.add_argument('--dropout_p', type=float, default=0.1)
    parser.add_argument('--num_kv_heads', type=int, default=None)
    parser.add_argument('--head_dim', type=int, default=None)
    parser.add_argument('--attn_dropout', type=float, default=0.1)
    parser.add_argument('--max_timestep', type=int, default=4096)
    parser.add_argument('--rope_theta', type=float, default=10_000.0)

    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--wt_decay', type=float, default=1e-4)
    parser.add_argument('--warmup_steps', type=int, default=10000)

    parser.add_argument('--max_train_iters', type=int, default=200)
    parser.add_argument('--num_updates_per_iter', type=int, default=100)

    parser.add_argument('--device', type=str, default='cuda')

    args = parser.parse_args()

    train(args)
