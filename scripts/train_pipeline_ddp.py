import argparse
import csv
import math
import os
import random
import time
import sys
from datetime import datetime
from typing import Dict, List, Optional, Sequence, Tuple

import gymnasium as gym
import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.distributed.pipelining import Schedule1F1B, build_stage, pipe_split, pipeline as make_pipeline
from torch.distributed.pipelining.microbatch import TensorChunkSpec
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

def _dbg(rank: int, message: str):
    print(f"[rank {rank}] {message}", flush=True)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from decision_transformer.pipeline import build_dt_pipeline_stages, build_qwen3_pipeline_stages
from decision_transformer.utils import D4RLTrajectoryDataset


def _create_env(env_name: str):
    try:
        return gym.make(env_name)
    except Exception as gymnasium_err:
        raise RuntimeError(
            f"Failed to construct environment '{env_name}' with gymnasium ({gymnasium_err}). "
            "Install the required environment packages such as 'mujoco' or 'gymnasium-robotics'."
        )


def _resolve_qwen_dims(args) -> Tuple[int, int, int, int]:
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


def _set_seed(seed: Optional[int]) -> None:
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _parse_device_groups(spec: str, expected_groups: int, expected_stages: int) -> List[List[int]]:
    if not spec:
        if expected_groups != 2 or expected_stages != 4:
            raise ValueError(
                "Default device groups only cover 2 data-parallel replicas with 4 pipeline stages each."
            )
        return [[0, 1, 2, 3], [4, 5, 6, 7]]

    groups: List[List[int]] = []
    for group_str in spec.split(";"):
        ids = [int(tok.strip()) for tok in group_str.split(",") if tok.strip()]
        if not ids:
            raise ValueError(f"Empty gpu id group in '{spec}'.")
        groups.append(ids)

    if len(groups) != expected_groups:
        raise ValueError(
            f"device_groups specifies {len(groups)} groups but --data_parallel_groups expected {expected_groups}."
        )

    for ids in groups:
        if len(ids) != expected_stages:
            raise ValueError(
                f"Each device group must contain {expected_stages} ids (pipeline stages). Got {ids}."
            )

    flat = [idx for grp in groups for idx in grp]
    if len(set(flat)) != len(flat):
        raise ValueError("GPU indices overlap between groups; each device must belong to exactly one stage.")

    available = torch.cuda.device_count()
    if any(idx < 0 or idx >= available for idx in flat):
        raise ValueError(
            f"device_groups references GPUs {sorted(set(flat))}, but only {available} CUDA devices are visible."
        )

    return groups


class SequentialPipelineModule(nn.Module):
    def __init__(self, stages: Sequence[nn.Module]):
        super().__init__()
        self.num_stages = len(stages)
        for idx, stage in enumerate(stages):
            self.add_module(f"stage_{idx}", stage)

    def forward(self, timesteps, states, actions, returns_to_go, traj_mask):
        output = getattr(self, "stage_0")((timesteps, states, actions, returns_to_go, traj_mask))
        for idx in range(1, self.num_stages):
            pipe_split()
            output = getattr(self, f"stage_{idx}")(output)
        return output


def _build_pipeline_blueprint(
    args,
    state_dim: int,
    act_dim: int,
) -> Tuple[SequentialPipelineModule, str, Dict[str, int]]:
    model_choice = args.model.lower()
    metadata: Dict[str, int] = {}

    if model_choice == "dt":
        stages, layout = build_dt_pipeline_stages(
            state_dim=state_dim,
            act_dim=act_dim,
            context_len=args.context_len,
            n_blocks=args.n_blocks,
            h_dim=args.embed_dim,
            n_heads=args.n_heads,
            drop_p=args.dropout_p,
            max_timestep=args.max_timestep,
            use_action_tanh=True,
            num_stages=args.pipeline_stages,
        )
        info = f"DecisionTransformer | hidden={args.embed_dim}, heads={args.n_heads}, per_stage={layout}"
    elif model_choice == "qwen3":
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
            use_action_tanh=True,
            num_stages=args.pipeline_stages,
        )
        info = (
            f"DecisionTransformerQwen3 | hidden={hidden_size}, heads={num_heads}, "
            f"kv_heads={num_kv_heads}, head_dim={head_dim}, per_stage={layout}"
        )
        metadata.update(
            dict(hidden_size=hidden_size, num_heads=num_heads, num_kv_heads=num_kv_heads, head_dim=head_dim)
        )
    else:
        raise ValueError(f"Unsupported model '{args.model}'. Expected 'dt' or 'qwen3'.")

    return SequentialPipelineModule(stages), info, metadata


def _broadcast_tensor(tensor: torch.Tensor, src_global_rank: int, group: dist.ProcessGroup) -> torch.Tensor:
    dist.broadcast(tensor, src=src_global_rank, group=group)
    return tensor


def _gather_stage_states(
    stage_state: Dict[str, torch.Tensor],
    pipeline_group: dist.ProcessGroup,
    stage_idx: int,
    is_group_root: bool,
) -> Optional[List[Dict[str, torch.Tensor]]]:
    cpu_state = {k: v.detach().cpu() for k, v in stage_state.items()}
    gather_list: Optional[List[Dict[str, torch.Tensor]]] = None
    if is_group_root:
        gather_list = [dict() for _ in range(dist.get_world_size(pipeline_group))]
    dist.gather_object(cpu_state, gather_list, dst=0, group=pipeline_group)
    if is_group_root:
        ordered: List[Dict[str, torch.Tensor]] = [dict() for _ in range(len(gather_list))]
        for rank_idx, payload in enumerate(gather_list):
            ordered[rank_idx] = payload
        return ordered
    return None


def train(args):
    dist.init_process_group(backend=args.dist_backend)
    world_size = dist.get_world_size()
    rank = dist.get_rank()

    if args.micro_batches < args.pipeline_stages:
        raise ValueError(
            f"1F1B schedule requires micro_batches >= pipeline_stages. Got {args.micro_batches} vs {args.pipeline_stages}."
        )
    _dbg(rank, "init_process_group complete")

    device_groups = _parse_device_groups(
        args.device_groups, expected_groups=args.data_parallel_groups, expected_stages=args.pipeline_stages
    )

    expected_world = len(device_groups) * args.pipeline_stages
    if world_size != expected_world:
        raise ValueError(
            f"torchrun world size ({world_size}) must equal data_parallel_groups * pipeline_stages = {expected_world}."
        )

    dp_world = len(device_groups)
    pipeline_group_ranks: List[List[int]] = []
    for dp_idx in range(dp_world):
        base = dp_idx * args.pipeline_stages
        pipeline_group_ranks.append([base + s for s in range(args.pipeline_stages)])

    stage_dp_group_ranks: List[List[int]] = []
    for stage_idx in range(args.pipeline_stages):
        stage_dp_group_ranks.append([dp_idx * args.pipeline_stages + stage_idx for dp_idx in range(dp_world)])

    dp_rank = rank // args.pipeline_stages
    pp_rank = rank % args.pipeline_stages
    _dbg(rank, f"dp_rank={dp_rank}, pp_rank={pp_rank}")
    
    time.sleep(dp_rank+4)
    pipeline_group = dist.new_group(ranks=pipeline_group_ranks[dp_rank])
    print(f"[{dist.get_rank()}] 第一个")
    time.sleep(dist.get_rank()+1)
    stage_dp_group = dist.new_group(ranks=stage_dp_group_ranks[pp_rank])
    print(f"[{dist.get_rank()}] 第二个")

    pipeline_group_rank = dist.get_rank(pipeline_group)
    stage_dp_group_size = dist.get_world_size(stage_dp_group)

    stage_devices = device_groups[dp_rank]
    device_idx = stage_devices[pp_rank]
    device = torch.device(f"cuda:{device_idx}")
    _dbg(rank, f"setting device {device}")
    torch.cuda.set_device(device)
    dist.barrier()
    _set_seed(args.seed + rank if args.seed is not None else None)
    _dbg(rank, f"device set to {device}")

    # Stage-0 specific setup: dataset + environment metadata
    env_arg = args.env.lower()

    if env_arg in {"walker2d", "walker2d-v3", "walker2d_v3"}:
        env_name = "Walker2d-v3"
        rtg_target = 5000.0
        env_d4rl_name = f"walker2d-{args.dataset}-v2"
    elif env_arg in {"halfcheetah", "halfcheetah-v3", "halfcheetah_v3"}:
        env_name = "HalfCheetah-v3"
        rtg_target = 6000.0
        env_d4rl_name = f"halfcheetah-{args.dataset}-v2"
    elif env_arg in {"hopper", "hopper-v3", "hopper_v3"}:
        env_name = "Hopper-v3"
        rtg_target = 3600.0
        env_d4rl_name = f"hopper-{args.dataset}-v2"
    elif env_arg in {"humanoid", "humanoid-v5", "humanoid_v5"}:
        env_name = "Humanoid-v5"
        env_d4rl_name = f"humanoid-{args.dataset}-v5"
        rtg_target = None
    else:
        raise NotImplementedError(f"Unknown env '{args.env}'.")

    env = None
    if pp_rank == 0:
        _dbg(rank, f"creating env {env_name}")
        env = _create_env(env_name)
        state_dim = env.observation_space.shape[0]
        act_dim = env.action_space.shape[0]
    else:
        state_dim = 0
        act_dim = 0
    dist.barrier()

    group_stage0_global = pipeline_group_ranks[dp_rank][0]
    group_stage_last_global = pipeline_group_ranks[dp_rank][-1]

    dims_tensor = torch.tensor([state_dim, act_dim], dtype=torch.int64, device=device)
    dims_tensor = _broadcast_tensor(dims_tensor, src_global_rank=group_stage0_global, group=pipeline_group)
    state_dim, act_dim = int(dims_tensor[0].item()), int(dims_tensor[1].item())

    dataset_path = os.path.join(args.dataset_dir, f"{env_d4rl_name}.pkl")

    if pp_rank == 0:
        _dbg(rank, "loading dataset")
        dataset = D4RLTrajectoryDataset(
            dataset_path,
            args.context_len,
            args.rtg_scale,
            minari_dataset=f"mujoco/humanoid/{args.dataset}-v0" if args.env == "humanoid" and not os.path.isfile(dataset_path) else None,
        )
        sampler = DistributedSampler(
            dataset,
            num_replicas=dp_world,
            rank=dp_rank,
            shuffle=True,
            drop_last=True,
        )
        data_loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            sampler=sampler,
            pin_memory=True,
            drop_last=True,
        )
        data_iter = iter(data_loader)
        state_mean, state_std = dataset.get_state_stats()
        if rtg_target is None:
            dataset_returns = np.array([np.sum(traj["rewards"]) for traj in dataset.trajectories])
            if dataset_returns.size == 0:
                raise ValueError("Dataset appears to be empty; cannot derive RTG target.")
            rtg_target = float(np.percentile(dataset_returns, 90))
    else:
        data_loader = None
        data_iter = None
        state_mean, state_std = np.zeros((state_dim,), dtype=np.float32), np.ones((state_dim,), dtype=np.float32)
    dist.barrier()

    rtg_tensor = torch.tensor([rtg_target if rtg_target is not None else 0.0], dtype=torch.float32, device=device)
    rtg_tensor = _broadcast_tensor(rtg_tensor, src_global_rank=group_stage0_global, group=pipeline_group)
    rtg_target = float(rtg_tensor.item())

    # Build pipeline blueprint and schedule
    _dbg(rank, "building pipeline blueprint")
    blueprint_module, model_info, _ = _build_pipeline_blueprint(args, state_dim, act_dim)
    micro_batch_size = args.batch_size // args.micro_batches

    example_inputs = (
        torch.zeros((micro_batch_size, args.context_len), dtype=torch.long),
        torch.zeros((micro_batch_size, args.context_len, state_dim), dtype=torch.float32),
        torch.zeros((micro_batch_size, args.context_len, act_dim), dtype=torch.float32),
        torch.zeros((micro_batch_size, args.context_len, 1), dtype=torch.float32),
        torch.ones((micro_batch_size, args.context_len), dtype=torch.float32),
    )

    _dbg(rank, "tracing pipeline with torch.distributed.pipelining")
    traced_pipe = make_pipeline(blueprint_module, example_inputs)
    dist.barrier()
    pipe_info = traced_pipe.info()
    stage_module = traced_pipe.get_stage_module(pp_rank)

    _dbg(rank, f"constructing pipeline stage {pp_rank}")
    stage = build_stage(
        stage_module=stage_module,
        stage_index=pp_rank,
        pipe_info=pipe_info,
        device=device,
        group=pipeline_group,
    )
    stage.submod.train()
    if stage_dp_group_size > 1:
        # import time
        # time.sleep(1)
        stage.submod = DDP(
            stage.submod,
            device_ids=[device_idx],
            process_group=stage_dp_group,
            broadcast_buffers=False,
            find_unused_parameters=False,
        )
    stage.submod.train()

    args_chunk_spec = TensorChunkSpec.from_tuple((0, 0, 0, 0, 0))
    output_merge_spec = (
        TensorChunkSpec(0),
        TensorChunkSpec(0),
        TensorChunkSpec(0),
        TensorChunkSpec(0),
    )

    def action_loss_fn(outputs, target):
        _, action_preds, _, traj_mask = outputs
        valid = traj_mask.reshape(-1) > 0.0
        preds_flat = action_preds.reshape(-1, act_dim)[valid]
        target_flat = target.reshape(-1, act_dim).to(action_preds.device)[valid]
        if preds_flat.numel() == 0:
            return torch.zeros((), device=action_preds.device, dtype=action_preds.dtype)
        return F.mse_loss(preds_flat, target_flat, reduction="mean")

    _dbg(rank, "creating Schedule1F1B")
    schedule = Schedule1F1B(
        stage=stage,
        n_microbatches=args.micro_batches,
        loss_fn=action_loss_fn,
        args_chunk_spec=args_chunk_spec,
        output_merge_spec=output_merge_spec,
        scale_grads=True,
    )

    optimizer = torch.optim.AdamW(stage.submod.parameters(), lr=args.lr, weight_decay=args.wt_decay)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda steps: min((steps + 1) / args.warmup_steps, 1.0))
    _dbg(rank, "optimizers ready")

    log_dir = args.log_dir
    os.makedirs(log_dir, exist_ok=True)

    start_time = datetime.now().replace(microsecond=0)
    start_time_str = start_time.strftime("%y-%m-%d-%H-%M-%S")

    model_tag = f"{args.model.lower()}pp"
    prefix = f"{model_tag}_{env_d4rl_name}"
    log_csv_path = os.path.join(log_dir, prefix + "_log_" + start_time_str + ".csv")
    save_model_path = os.path.join(log_dir, prefix + "_model_" + start_time_str + ".pt")

    csv_file = None
    csv_writer = None
    if rank == 0:
        csv_file = open(log_csv_path, "a", buffering=1)
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(["duration", "num_updates", "action_loss"])
        print("=" * 60)
        print("Distributed pipeline training configuration")
        print("=" * 60)
        print(f"Start time: {start_time_str}")
        print(f"Model info: {model_info}")
        print(f"Pipeline stages: {args.pipeline_stages}")
        print(f"Data parallel groups: {dp_world}")
        print(f"Micro batches: {args.micro_batches}")
        print(f"Device groups: {device_groups}")

    total_updates = 0

    for iter_idx in range(args.max_train_iters):
        if pp_rank == 0 and data_loader is not None:
            sampler.set_epoch(iter_idx)

        stage_loss_values: List[float] = []
        for _ in range(args.num_updates_per_iter):
            if pp_rank == 0:
                try:
                    batch = next(data_iter)
                except StopIteration:
                    data_iter = iter(data_loader)  # type: ignore[arg-type]
                    batch = next(data_iter)

                timesteps, states, actions, returns_to_go, traj_mask = batch

                timesteps = timesteps.to(device=device, dtype=torch.long, non_blocking=True)
                states = states.to(device=device, dtype=torch.float32, non_blocking=True)
                actions = actions.to(device=device, dtype=torch.float32, non_blocking=True)
                returns_to_go = returns_to_go.to(device=device, dtype=torch.float32, non_blocking=True).unsqueeze(-1)
                traj_mask = traj_mask.to(device=device, dtype=torch.float32, non_blocking=True)
            else:
                timesteps = torch.empty((args.batch_size, args.context_len), dtype=torch.long, device=device)
                states = torch.empty((args.batch_size, args.context_len, state_dim), dtype=torch.float32, device=device)
                actions = torch.empty((args.batch_size, args.context_len, act_dim), dtype=torch.float32, device=device)
                returns_to_go = torch.empty((args.batch_size, args.context_len, 1), dtype=torch.float32, device=device)
                traj_mask = torch.empty((args.batch_size, args.context_len), dtype=torch.float32, device=device)

            _dbg(rank, "broadcasting batch tensors")
            _broadcast_tensor(timesteps, src_global_rank=group_stage0_global, group=pipeline_group)
            _broadcast_tensor(states, src_global_rank=group_stage0_global, group=pipeline_group)
            _broadcast_tensor(actions, src_global_rank=group_stage0_global, group=pipeline_group)
            _broadcast_tensor(returns_to_go, src_global_rank=group_stage0_global, group=pipeline_group)
            _broadcast_tensor(traj_mask, src_global_rank=group_stage0_global, group=pipeline_group)

            action_target = actions.clone()

            optimizer.zero_grad(set_to_none=True)
            losses: List[torch.Tensor] = []
            _dbg(rank, "running schedule step")
            schedule.step(
                timesteps,
                states,
                actions,
                returns_to_go,
                traj_mask,
                target=action_target,
                losses=losses,
            )

            if args.grad_clip is not None and args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(stage.submod.parameters(), args.grad_clip)
            optimizer.step()
            scheduler.step()

            loss_tensor = torch.zeros(1, device=device)
            if pp_rank == args.pipeline_stages - 1:
                if losses:
                    stacked = torch.stack([loss.detach() for loss in losses])
                    loss_tensor = stacked.mean()
                else:
                    loss_tensor = torch.zeros(1, device=device)
            _broadcast_tensor(loss_tensor, src_global_rank=group_stage_last_global, group=pipeline_group)

            if pp_rank == 0:
                aggregate = loss_tensor.clone()
                dist.all_reduce(aggregate, op=dist.ReduceOp.SUM, group=stage_dp_group)
                aggregate /= float(stage_dp_group_size)
                if rank == pipeline_group_ranks[0][0]:
                    stage_loss_values.append(float(aggregate.item()))

            total_updates += args.micro_batches

        if pp_rank == 0 and rank == 0:
            mean_loss = float(np.mean(stage_loss_values)) if stage_loss_values else float("nan")
            time_elapsed = str(datetime.now().replace(microsecond=0) - start_time)
            print("=" * 60)
            print(f"time elapsed: {time_elapsed}")
            print(f"num of updates: {total_updates}")
            print(f"action loss: {mean_loss:.5f}")
            csv_writer.writerow([time_elapsed, total_updates, mean_loss])

    # Save final weights (gather stage states to pipeline group root then global rank 0 saves)
    state_dict = stage.submod.module.state_dict() if isinstance(stage.submod, DDP) else stage.submod.state_dict()
    stage_states = _gather_stage_states(
        state_dict,
        pipeline_group=pipeline_group,
        stage_idx=pp_rank,
        is_group_root=(pipeline_group_rank == 0),
    )

    if pipeline_group_rank == 0:
        for local_idx, state in enumerate(stage_states or []):
            traced_pipe.get_stage_module(local_idx).load_state_dict(state)

    dist.barrier()

    if rank == 0:
        torch.save(traced_pipe.split_gm.state_dict(), save_model_path)
        end_time = datetime.now().replace(microsecond=0)
        time_elapsed = str(end_time - start_time)
        end_time_str = end_time.strftime("%y-%m-%d-%H-%M-%S")
        print("=" * 60)
        print("finished training!")
        print("=" * 60)
        print(f"started training at: {start_time_str}")
        print(f"finished training at: {end_time_str}")
        print(f"total training time: {time_elapsed}")
        print(f"saved pipeline state dict at: {save_model_path}")
        print("=" * 60)
        csv_file.close()  # type: ignore[union-attr]

    dist.barrier()
    dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description="Hybrid pipeline + data parallel training using torch.distributed.pipelining.")

    parser.add_argument("--env", type=str, default="halfcheetah")
    parser.add_argument("--dataset", type=str, default="medium")
    parser.add_argument("--rtg_scale", type=int, default=1000)

    parser.add_argument("--max_eval_ep_len", type=int, default=1000)  # retained for compatibility
    parser.add_argument("--num_eval_ep", type=int, default=10)

    parser.add_argument("--dataset_dir", type=str, default="data/")
    parser.add_argument("--log_dir", type=str, default="dt_runs/")

    parser.add_argument("--model", type=str, default="qwen3", choices=["dt", "qwen3"])
    parser.add_argument("--context_len", type=int, default=20)
    parser.add_argument("--n_blocks", type=int, default=12)
    parser.add_argument("--embed_dim", type=int, default=512)
    parser.add_argument("--n_heads", type=int, default=8)
    parser.add_argument("--dropout_p", type=float, default=0.1)
    parser.add_argument("--num_kv_heads", type=int, default=None)
    parser.add_argument("--head_dim", type=int, default=None)
    parser.add_argument("--attn_dropout", type=float, default=0.1)
    parser.add_argument("--max_timestep", type=int, default=4096)
    parser.add_argument("--rope_theta", type=float, default=10_000.0)

    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--micro_batches", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--wt_decay", type=float, default=1e-4)
    parser.add_argument("--warmup_steps", type=int, default=10000)
    parser.add_argument("--grad_clip", type=float, default=0.25)

    parser.add_argument("--max_train_iters", type=int, default=200)
    parser.add_argument("--num_updates_per_iter", type=int, default=100)

    parser.add_argument("--pipeline_stages", type=int, default=4)
    parser.add_argument("--data_parallel_groups", type=int, default=2)
    parser.add_argument(
        "--device_groups",
        type=str,
        default=None,
        help="Semicolon separated GPU ids for each pipeline group, e.g. '0,1,2,3;4,5,6,7'.",
    )

    parser.add_argument("--dist_backend", type=str, default="nccl")
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    if args.batch_size % args.micro_batches != 0:
        raise ValueError("batch_size must be divisible by micro_batches for pipeline parallelism.")

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for pipeline training.")

    train(args)


if __name__ == "__main__":
    main()
