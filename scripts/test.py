import argparse
import os
import sys
import math
import gymnasium as gym
import torch
import torch.nn as nn
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from decision_transformer.utils import (
    evaluate_on_env,
    get_d4rl_normalized_score,
    get_d4rl_dataset_stats,
    D4RLTrajectoryDataset,
)
from decision_transformer.model import DecisionTransformer, DecisionTransformerQwen3
from decision_transformer.pipeline import (
    build_dt_pipeline_stages,
    build_qwen3_pipeline_stages,
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


class SequentialPipelineModule(nn.Module):
    def __init__(self, stages):
        super().__init__()
        self.num_stages = len(stages)
        for idx, stage in enumerate(stages):
            self.add_module(f"stage_{idx}", stage)

    def forward(self, timesteps, states, actions, returns_to_go, traj_mask):
        output = getattr(self, "stage_0")((timesteps, states, actions, returns_to_go, traj_mask))
        for idx in range(1, self.num_stages):
            stage = getattr(self, f"stage_{idx}")
            output = stage(output)
        return output


def _build_pipeline_model(args, state_dim, act_dim, device):
    model_choice = args.model.lower()
    use_action_tanh = True
    if model_choice == 'dt':
        stages, layout = build_dt_pipeline_stages(
            state_dim=state_dim,
            act_dim=act_dim,
            context_len=args.context_len,
            n_blocks=args.n_blocks,
            h_dim=args.embed_dim,
            n_heads=args.n_heads,
            drop_p=args.dropout_p,
            max_timestep=args.max_timestep,
            use_action_tanh=use_action_tanh,
            num_stages=args.pipeline_stages,
        )
        info = f"DT pipeline | hidden={args.embed_dim}, heads={args.n_heads}, per_stage={layout}"
    elif model_choice == 'qwen3':
        hidden_size, num_heads, num_kv_heads, head_dim = _resolve_qwen_dims(args)
        stages, layout = build_qwen3_pipeline_stages(
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
            rope_theta=args.rope_theta,
            max_timestep=args.max_timestep,
            use_action_tanh=use_action_tanh,
            num_stages=args.pipeline_stages,
        )
        info = (
            f"Qwen3 pipeline | hidden={hidden_size}, heads={num_heads}, "
            f"kv_heads={num_kv_heads}, head_dim={head_dim}, per_stage={layout}"
        )
    else:
        raise ValueError(f"Unsupported pipeline model '{args.model}'.")
    return SequentialPipelineModule(stages).to(device), info


def _detect_checkpoint_format(state_dict, override):
    if override != "auto":
        return override
    for key in state_dict.keys():
        if key.startswith("submod_") or key.startswith("stage_0"):
            return "pipeline"
    return "standard"


def _extract_pipeline_state_dict(state_dict):
    cleaned = {}
    for key, value in state_dict.items():
        new_key = key
        if key.startswith("submod_"):
            parts = key.split(".", 2)
            if len(parts) >= 2 and parts[1].startswith("stage_"):
                new_key = parts[1]
                if len(parts) == 3:
                    new_key = f"{new_key}.{parts[2]}"
        if new_key.startswith("stage_"):
            cleaned[new_key] = value
    if not cleaned:
        raise RuntimeError(
            "Checkpoint appears to use pipeline format, but no 'stage_' parameters were found."
        )
    return cleaned


def _create_env(env_name, render=False):
    render_kwargs = {"render_mode": "human"} if render else {}

    try:
        return gym.make(env_name, **render_kwargs)
    except TypeError:
        # Some Gymnasium envs might not support render_mode.
        if render_kwargs:
            return gym.make(env_name)
        raise
    except Exception as gymnasium_err:
        raise RuntimeError(
            f"Failed to construct environment '{env_name}' with gymnasium ({gymnasium_err}). "
            "Ensure all required plugins (e.g., mujoco, gymnasium-robotics) are installed."
        )

def test(args):

    eval_dataset = args.dataset         # medium / medium-replay / medium-expert
    eval_rtg_scale = args.rtg_scale     # normalize returns to go
    dataset_dir = args.dataset_dir

    minari_dataset = None
    eval_rtg_target = None

    if args.env == 'walker2d':
        eval_env_name = 'Walker2d-v3'
        eval_rtg_target = 5000
        eval_env_d4rl_name = f'walker2d-{eval_dataset}-v2'
        dataset_path = f'{dataset_dir}/{eval_env_d4rl_name}.pkl'

    elif args.env == 'halfcheetah':
        eval_env_name = 'HalfCheetah-v3'
        eval_rtg_target = 6000
        eval_env_d4rl_name = f'halfcheetah-{eval_dataset}-v2'
        dataset_path = f'{dataset_dir}/{eval_env_d4rl_name}.pkl'

    elif args.env == 'hopper':
        eval_env_name = 'Hopper-v3'
        eval_rtg_target = 3600
        eval_env_d4rl_name = f'hopper-{eval_dataset}-v2'
        dataset_path = f'{dataset_dir}/{eval_env_d4rl_name}.pkl'

    elif args.env == 'humanoid':
        eval_env_name = 'Humanoid-v5'
        eval_env_d4rl_name = f'humanoid-{eval_dataset}-v5'
        dataset_path = f'{dataset_dir}/{eval_env_d4rl_name}.pkl'
        minari_dataset = f'mujoco/humanoid/{eval_dataset}-v0'

    else:
        raise NotImplementedError

    render = args.render                # render the env frames

    num_test_eval_ep = args.num_eval_ep         # num of evaluation episodes
    eval_max_eval_ep_len = args.max_eval_ep_len # max len of one episode

    context_len = args.context_len      # K in decision transformer


    eval_chk_pt_dir = args.chk_pt_dir

    eval_chk_pt_name = args.chk_pt_name
    eval_chk_pt_list = [eval_chk_pt_name]


    ## manually override check point list
    ## passing a list will evaluate on all checkpoints
    ## and output mean and std score

    # eval_chk_pt_list = [
    #     "dt_halfcheetah-medium-v2_model_22-02-09-10-38-54_best.pt",
    #     "dt_halfcheetah-medium-v2_model_22-02-10-11-56-32_best.pt",
    #     "dt_halfcheetah-medium-v2_model_22-02-11-10-13-57_best.pt"
    # ]


    device = torch.device(args.device)
    model_choice = args.model.lower()
    print("device set to: ", device)
    print("model architecture: ", model_choice)

    if args.env == 'humanoid':
        traj_dataset = D4RLTrajectoryDataset(
                            dataset_path,
                            args.context_len,
                            eval_rtg_scale,
                            minari_dataset=minari_dataset
                        )
        eval_state_mean, eval_state_std = traj_dataset.get_state_stats()
        dataset_returns = np.array([np.sum(traj['rewards']) for traj in traj_dataset.trajectories])
        if dataset_returns.size == 0:
            raise ValueError('Humanoid dataset is empty; cannot evaluate model.')
        eval_rtg_target = float(np.percentile(dataset_returns, 90)) if eval_rtg_target is None else eval_rtg_target
        print(f"derived humanoid rtg target (p90 dataset return): {eval_rtg_target:.2f}")
        dataset_source = dataset_path if os.path.isfile(dataset_path) else f"minari:{minari_dataset}"
        print("dataset source: " + dataset_source)
    else:
        env_data_stats = get_d4rl_dataset_stats(eval_env_d4rl_name)
        eval_state_mean = np.array(env_data_stats['state_mean'])
        eval_state_std = np.array(env_data_stats['state_std'])

    eval_env = _create_env(eval_env_name, render)

    state_dim = eval_env.observation_space.shape[0]
    act_dim = eval_env.action_space.shape[0]

    all_scores = []

    for idx, eval_chk_pt_name in enumerate(eval_chk_pt_list):

        eval_chk_pt_path = os.path.join(eval_chk_pt_dir, eval_chk_pt_name)
        checkpoint_state = torch.load(eval_chk_pt_path, map_location="cpu")
        ckpt_format = _detect_checkpoint_format(checkpoint_state, args.ckpt_format)

        if ckpt_format == "pipeline":
            eval_model, model_info = _build_pipeline_model(args, state_dim, act_dim, device)
            cleaned_state = _extract_pipeline_state_dict(checkpoint_state)
            eval_model.load_state_dict(cleaned_state, strict=True)
        else:
            eval_model, model_info = _build_model(args, state_dim, act_dim, device)
            eval_model.load_state_dict(checkpoint_state, strict=True)

        del checkpoint_state

        if idx == 0:
            print(f"checkpoint format: {ckpt_format}")
            print("model resolved dims: " + model_info)
            param_total = sum(p.numel() for p in eval_model.parameters())
            print(f"model parameter count: {param_total / 1e6:.3f}M")

        print("model loaded from: " + eval_chk_pt_path)

        # evaluate on env
        results = evaluate_on_env(eval_model, device, context_len,
                                eval_env, eval_rtg_target, eval_rtg_scale,
                                num_test_eval_ep, eval_max_eval_ep_len,
                                eval_state_mean, eval_state_std, render=render)
        print(results)

        norm_score = get_d4rl_normalized_score(results['eval/avg_reward'], eval_env_name) * 100
        if math.isnan(norm_score):
            print("normalized d4rl score: nan (reference stats unavailable)")
        else:
            print("normalized d4rl score: " + format(norm_score, ".5f"))

        all_scores.append(norm_score)

    print("=" * 60)
    all_scores = np.array(all_scores)
    print("evaluated on env: " + eval_env_name)
    print("total num of checkpoints evaluated: " + str(len(eval_chk_pt_list)))
    print("d4rl score mean: " + format(all_scores.mean(), ".5f"))
    print("d4rl score std: " + format(all_scores.std(), ".5f"))
    print("d4rl score var: " + format(all_scores.var(), ".5f"))
    print("=" * 60)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('--env', type=str, default='halfcheetah')
    parser.add_argument('--dataset', type=str, default='medium')
    parser.add_argument('--rtg_scale', type=int, default=1000)

    parser.add_argument('--max_eval_ep_len', type=int, default=1000)
    parser.add_argument('--num_eval_ep', type=int, default=10)

    parser.add_argument("--render", action="store_true", default=False)

    parser.add_argument('--chk_pt_dir', type=str, default='dt_runs/')
    parser.add_argument('--chk_pt_name', type=str,
            default='dt_halfcheetah-medium-v2_model_22-02-13-09-03-10_best.pt')

    parser.add_argument('--dataset_dir', type=str, default='data/')

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
    parser.add_argument('--pipeline_stages', type=int, default=4,
            help='Number of pipeline stages to use when reconstructing pipeline checkpoints.')
    parser.add_argument('--ckpt_format', type=str, default='auto', choices=['auto', 'standard', 'pipeline'],
            help="Force checkpoint format detection. Use 'pipeline' for checkpoints saved by train_pipeline_ddp.py.")

    parser.add_argument('--device', type=str, default='cuda')

    args = parser.parse_args()

    test(args)
