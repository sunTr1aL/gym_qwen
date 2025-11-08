import argparse
import csv
import math
import os
import random
import re
import time
import sys
from contextlib import contextmanager
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


class ProgressTracker:
    def __init__(self, total_steps: int, refresh_interval_s: float = 2.0):
        self.total = max(1, total_steps)
        self.refresh = max(0.1, refresh_interval_s)
        self.start = time.time()
        self.last_render = self.start
        self.completed = 0
        self._last_bar = ""

    def update(self, delta: int, force: bool = False) -> None:
        self.completed = min(self.total, self.completed + max(0, delta))
        now = time.time()
        if force or (now - self.last_render) >= self.refresh or self.completed == self.total:
            self._render(now)

    def _render(self, now: float) -> None:
        elapsed = max(now - self.start, 1e-6)
        speed = self.completed / elapsed
        ratio = self.completed / self.total
        bar_len = 30
        filled = int(bar_len * ratio)
        bar = "#" * filled + "-" * (bar_len - filled)
        remaining = self.total - self.completed
        eta = remaining / speed if speed > 0 else float("inf")
        eta_str = "N/A" if not math.isfinite(eta) else time.strftime("%H:%M:%S", time.gmtime(eta))
        msg = (
            f"\r[progress] |{bar}| {ratio*100:6.2f}% "
            f"{self.completed}/{self.total} updates | {speed:6.2f} upd/s | ETA {eta_str}"
        )
        print(msg, end="", flush=True)
        self._last_bar = msg
        self.last_render = now

    def close(self) -> None:
        if self._last_bar:
            print()
            self._last_bar = ""

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from decision_transformer.pipeline import build_dt_pipeline_stages, build_qwen3_pipeline_stages
from decision_transformer.utils import D4RLTrajectoryDataset, evaluate_on_env


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
    gather_group: dist.ProcessGroup,
    stage_idx: int,
    is_group_root: bool,
    group_root_global: int,
) -> Optional[List[Dict[str, torch.Tensor]]]:
    cpu_state = {k: v.detach().cpu() for k, v in stage_state.items()}
    gather_list: Optional[List[Dict[str, torch.Tensor]]] = None
    if is_group_root:
        gather_list = [dict() for _ in range(dist.get_world_size(gather_group))]
    dist.gather_object(cpu_state, gather_list, dst=group_root_global, group=gather_group)
    if is_group_root:
        ordered: List[Dict[str, torch.Tensor]] = [dict() for _ in range(len(gather_list))]
        for rank_idx, payload in enumerate(gather_list):
            ordered[rank_idx] = payload
        return ordered
    return None


def _save_pipeline_checkpoint(
    *,
    traced_pipe,
    stage_module_actual: nn.Module,
    gather_group: dist.ProcessGroup,
    pipeline_group_rank: int,
    group_stage0_global: int,
    pp_rank: int,
    save_path: str,
    sync_group: Optional[dist.ProcessGroup] = None,
) -> None:
    if sync_group is None:
        sync_group = dist.group.WORLD
    state_dict = stage_module_actual.state_dict()
    stage_states = _gather_stage_states(
        state_dict,
        gather_group=gather_group,
        stage_idx=pp_rank,
        is_group_root=(pipeline_group_rank == 0),
        group_root_global=group_stage0_global,
    )
    if pipeline_group_rank == 0:
        for local_idx, state in enumerate(stage_states or []):
            traced_pipe.get_stage_module(local_idx).load_state_dict(state)
    dist.barrier(group=sync_group)
    rank = dist.get_rank()
    if rank == 0:
        torch.save(traced_pipe.split_gm.state_dict(), save_path)
        _dbg(rank, f"saved checkpoint to {save_path}")
    dist.barrier(group=sync_group)


def _load_pipeline_checkpoint(
    *,
    traced_pipe,
    stage_module_actual: nn.Module,
    pp_rank: int,
    checkpoint_path: str,
    strict: bool = True,
    sync_group: Optional[dist.ProcessGroup] = None,
) -> None:
    if sync_group is None:
        sync_group = dist.group.WORLD
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint '{checkpoint_path}' not found.")
    rank = dist.get_rank()
    if rank == 0:
        _dbg(rank, f"loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    missing, unexpected = traced_pipe.split_gm.load_state_dict(checkpoint, strict=False)
    if strict and (missing or unexpected):
        raise RuntimeError(
            f"Checkpoint mismatch while loading '{checkpoint_path}': missing={missing}, unexpected={unexpected}"
        )
    stage_state = traced_pipe.get_stage_module(pp_rank).state_dict()
    stage_module_actual.load_state_dict(stage_state, strict=strict)
    stage_module_actual.train()
    dist.barrier(group=sync_group)


def _infer_resume_progress_from_name(path: str) -> Tuple[Optional[int], Optional[int]]:
    """Extract completed iterations and total updates from checkpoint file name."""
    base = os.path.basename(path)
    iter_match = re.search(r"iter(\d+)", base)
    updates_match = re.search(r"updates(\d+)", base)
    iter_idx = int(iter_match.group(1)) if iter_match else None
    updates = int(updates_match.group(1)) if updates_match else None
    return iter_idx, updates


def _strip_stage_prefix(stage_idx: int, state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Remove the leading 'stage_{idx}.' prefix applied by FX graph modules."""
    prefix = f"stage_{stage_idx}."
    cleaned: Dict[str, torch.Tensor] = {}
    for key, tensor in state.items():
        new_key = key[len(prefix):] if key.startswith(prefix) else key
        cleaned[new_key] = tensor
    return cleaned


def _move_optimizer_state(optimizer: Optional[torch.optim.Optimizer], device: torch.device) -> None:
    if optimizer is None:
        return
    for state in optimizer.state.values():
        for key, value in state.items():
            if torch.is_tensor(value):
                state[key] = value.to(device, non_blocking=True)


def _empty_cuda_cache(device: torch.device) -> None:
    if device.type != "cuda":
        return
    with torch.cuda.device(device):
        torch.cuda.empty_cache()


@contextmanager
def _temporarily_offload_module(
    module: nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    device: torch.device,
) -> None:
    if device.type != "cuda":
        yield
        return
    cpu_device = torch.device("cpu")
    module.to(cpu_device)
    _move_optimizer_state(optimizer, cpu_device)
    _empty_cuda_cache(device)
    try:
        yield
    finally:
        module.to(device)
        _move_optimizer_state(optimizer, device)
        module.train()
        _empty_cuda_cache(device)


def train(args):
    dist.init_process_group(backend=args.dist_backend)
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    global_sync_group = dist.new_group(ranks=list(range(world_size)), backend="gloo")

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
    
    all_pipeline_groups = []
    all_pipeline_groups_cpu = []
    for grp in pipeline_group_ranks:
        g = dist.new_group(ranks=grp)
        all_pipeline_groups.append(g)
        g_cpu = dist.new_group(ranks=grp, backend="gloo")
        all_pipeline_groups_cpu.append(g_cpu)
    pipeline_group = all_pipeline_groups[dp_rank]
    pipeline_group_cpu = all_pipeline_groups_cpu[dp_rank]
    print(f"[{dist.get_rank()}] 第一个 {pipeline_group_ranks[dp_rank]}")
    all_stage_dp_groups = []
    for grp in stage_dp_group_ranks:
        g = dist.new_group(ranks=grp)
        all_stage_dp_groups.append(g)
    stage_dp_group = all_stage_dp_groups[pp_rank]
    print(f"[{dist.get_rank()}] 第二个 {stage_dp_group_ranks[pp_rank]}")


    pipeline_group_rank = dist.get_rank(pipeline_group)
    stage_dp_group_size = dist.get_world_size(stage_dp_group)

    stage_devices = device_groups[dp_rank]
    device_idx = stage_devices[pp_rank]
    device = torch.device(f"cuda:{device_idx}")
    _dbg(rank, f"setting device {device}")
    torch.cuda.set_device(device)
    dist.barrier(group=global_sync_group)
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
    dist.barrier(group=global_sync_group)

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
    dist.barrier(group=global_sync_group)

    rtg_tensor = torch.tensor([rtg_target if rtg_target is not None else 0.0], dtype=torch.float32, device=device)
    rtg_tensor = _broadcast_tensor(rtg_tensor, src_global_rank=group_stage0_global, group=pipeline_group)
    rtg_target = float(rtg_tensor.item())

    # Build pipeline blueprint and schedule
    _dbg(rank, "building pipeline blueprint")
    blueprint_module, model_info, _ = _build_pipeline_blueprint(args, state_dim, act_dim)
    blueprint_module = blueprint_module.to(device)
    micro_batch_size = args.batch_size // args.micro_batches
    eval_model: Optional[SequentialPipelineModule] = None

    example_inputs = (
        torch.zeros((micro_batch_size, args.context_len), dtype=torch.long, device=device),
        torch.zeros((micro_batch_size, args.context_len, state_dim), dtype=torch.float32, device=device),
        torch.zeros((micro_batch_size, args.context_len, act_dim), dtype=torch.float32, device=device),
        torch.zeros((micro_batch_size, args.context_len, 1), dtype=torch.float32, device=device),
        torch.zeros((micro_batch_size, args.context_len), dtype=torch.float32, device=device),
    )

    _dbg(rank, "tracing pipeline with torch.distributed.pipelining")
    traced_pipe = make_pipeline(blueprint_module, example_inputs)
    dist.barrier(group=global_sync_group)
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
    ddp_wrapper = None
    if stage_dp_group_size > 1:
        ddp_wrapper = DDP(
            stage.submod,
            device_ids=[device_idx],
            process_group=stage_dp_group,
            broadcast_buffers=False,
            find_unused_parameters=False,
        )
    stage_module_actual = ddp_wrapper.module if ddp_wrapper is not None else stage.submod
    stage_module_actual.train()

    args_chunk_spec = TensorChunkSpec.from_tuple((0, 0, 0, 0, 0))
    output_merge_spec = None

    def action_loss_fn(outputs, target):
        traj_mask = target[..., -1]
        action_target = target[..., :-1]
        _, action_preds, _ = outputs
        valid = traj_mask.reshape(-1) > 0.0
        preds_flat = action_preds.reshape(-1, act_dim)[valid]
        target_flat = action_target.reshape(-1, act_dim)[valid]
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
    )

    optimizer = torch.optim.AdamW(stage.submod.parameters(), lr=args.lr, weight_decay=args.wt_decay)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda steps: min((steps + 1) / args.warmup_steps, 1.0))
    _dbg(rank, "optimizers ready")

    resume_iter_idx = 0
    resume_updates = 0
    if args.resume_checkpoint:
        _load_pipeline_checkpoint(
            traced_pipe=traced_pipe,
            stage_module_actual=stage_module_actual,
            pp_rank=pp_rank,
            checkpoint_path=args.resume_checkpoint,
            sync_group=global_sync_group,
        )
        inferred_iter, inferred_updates = _infer_resume_progress_from_name(args.resume_checkpoint)
        if args.resume_iter is not None:
            resume_iter_idx = max(0, args.resume_iter)
        elif inferred_iter is not None:
            resume_iter_idx = max(0, inferred_iter)
        if args.resume_updates is not None:
            resume_updates = max(0, args.resume_updates)
        elif inferred_updates is not None:
            resume_updates = max(0, inferred_updates)
        if rank == 0:
            print(
                f"Resumed pipeline weights from {args.resume_checkpoint} "
                f"(start_iter={resume_iter_idx}, total_updates={resume_updates})"
            )
    start_iter_idx = min(resume_iter_idx, args.max_train_iters)
    remaining_iters = max(args.max_train_iters - start_iter_idx, 0)
    total_progress_updates = max(remaining_iters * args.num_updates_per_iter, 1)
    progress_tracker = (
        ProgressTracker(total_progress_updates, args.progress_refresh) if rank == 0 and remaining_iters > 0 else None
    )
    latest_eval_reward = float("nan")
    latest_eval_ep_len = float("nan")

    def _evaluate_policy_if_needed() -> Tuple[Optional[float], Optional[float]]:
        nonlocal eval_model
        stage_state = stage_module_actual.state_dict()
        stage_states = _gather_stage_states(
            stage_state,
            gather_group=pipeline_group_cpu,
            stage_idx=pp_rank,
            is_group_root=(pipeline_group_rank == 0),
            group_root_global=group_stage0_global,
        )
        eval_reward: Optional[float] = None
        eval_ep_len: Optional[float] = None
        should_run_eval = stage_states is not None and rank == pipeline_group_ranks[0][0] and env is not None
        if should_run_eval:
            if eval_model is None:
                eval_model, _, _ = _build_pipeline_blueprint(args, state_dim, act_dim)
                eval_model = eval_model.to(torch.device("cpu"))
            for local_idx, state in enumerate(stage_states or []):
                target_module = getattr(eval_model, f"stage_{local_idx}", None)
                if target_module is None:
                    continue
                cleaned = _strip_stage_prefix(local_idx, state)
                target_module.load_state_dict(cleaned, strict=False)
            with _temporarily_offload_module(stage_module_actual, optimizer, device):
                eval_model = eval_model.to(device)
                _empty_cuda_cache(device)
                eval_model.eval()
                with torch.no_grad():
                    results = evaluate_on_env(
                        eval_model,
                        device,
                        args.context_len,
                        env,
                        rtg_target,
                        args.rtg_scale,
                        args.num_eval_ep,
                        args.max_eval_ep_len,
                        state_mean=state_mean,
                        state_std=state_std,
                    )
                eval_reward = float(results["eval/avg_reward"])
                eval_ep_len = float(results["eval/avg_ep_len"])
                eval_model = eval_model.to(torch.device("cpu"))
                _empty_cuda_cache(device)
        dist.barrier(group=global_sync_group)
        return eval_reward, eval_ep_len

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
        csv_writer.writerow(["duration", "num_updates", "action_loss", "eval_avg_reward", "eval_avg_ep_len"])
        print("=" * 60)
        print("Distributed pipeline training configuration")
        print("=" * 60)
        print(f"Start time: {start_time_str}")
        print(f"Model info: {model_info}")
        print(f"Pipeline stages: {args.pipeline_stages}")
        print(f"Data parallel groups: {dp_world}")
        print(f"Micro batches: {args.micro_batches}")
        print(f"Device groups: {device_groups}")

    total_updates = resume_updates

    for iter_idx in range(start_iter_idx, args.max_train_iters):
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

            if args.log_stage_steps:
                _dbg(rank, "broadcasting batch tensors")
            _broadcast_tensor(timesteps, src_global_rank=group_stage0_global, group=pipeline_group)
            _broadcast_tensor(states, src_global_rank=group_stage0_global, group=pipeline_group)
            _broadcast_tensor(actions, src_global_rank=group_stage0_global, group=pipeline_group)
            _broadcast_tensor(returns_to_go, src_global_rank=group_stage0_global, group=pipeline_group)
            _broadcast_tensor(traj_mask, src_global_rank=group_stage0_global, group=pipeline_group)

            action_target = torch.cat(
                (actions.detach(), traj_mask.unsqueeze(-1)),
                dim=-1,
            ).detach()

            optimizer.zero_grad(set_to_none=True)
            losses: List[torch.Tensor] = []
            if args.log_stage_steps:
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
                torch.nn.utils.clip_grad_norm_(stage_module_actual.parameters(), args.grad_clip)
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
                    if progress_tracker is not None:
                        progress_tracker.update(1)

            total_updates += 1

        if pp_rank == 0 and rank == 0:
            mean_loss = float(np.mean(stage_loss_values)) if stage_loss_values else float("nan")
            time_elapsed = str(datetime.now().replace(microsecond=0) - start_time)
            if progress_tracker is not None:
                progress_tracker.update(0, force=True)
            print("=" * 60)
            print(f"time elapsed: {time_elapsed}")
            print(f"num of updates: {total_updates}")
            print(f"action loss: {mean_loss:.5f}")
            csv_writer.writerow([time_elapsed, total_updates, mean_loss, latest_eval_reward, latest_eval_ep_len])

        should_eval = args.eval_interval > 0 and (
            (iter_idx + 1) % args.eval_interval == 0 or iter_idx == args.max_train_iters - 1
        )
        if should_eval:
            eval_reward, eval_ep_len = _evaluate_policy_if_needed()
            if eval_reward is not None:
                latest_eval_reward = eval_reward
            if eval_ep_len is not None:
                latest_eval_ep_len = eval_ep_len
            if rank == pipeline_group_ranks[0][0] and eval_reward is not None:
                print(
                    f"[eval] iter={iter_idx + 1} "
                    f"reward={latest_eval_reward:.3f} "
                    f"ep_len={latest_eval_ep_len:.2f}"
                )
            checkpoint_path = os.path.join(
                log_dir,
                f"{prefix}_ckpt_iter{iter_idx + 1:05d}_updates{total_updates:08d}.pt",
            )
            _save_pipeline_checkpoint(
                traced_pipe=traced_pipe,
                stage_module_actual=stage_module_actual,
                gather_group=pipeline_group_cpu,
                pipeline_group_rank=pipeline_group_rank,
                group_stage0_global=group_stage0_global,
                pp_rank=pp_rank,
                save_path=checkpoint_path,
                sync_group=global_sync_group,
            )

    _save_pipeline_checkpoint(
        traced_pipe=traced_pipe,
        stage_module_actual=stage_module_actual,
        gather_group=pipeline_group_cpu,
        pipeline_group_rank=pipeline_group_rank,
        group_stage0_global=group_stage0_global,
        pp_rank=pp_rank,
        save_path=save_model_path,
        sync_group=global_sync_group,
    )

    if rank == 0:
        if progress_tracker is not None:
            progress_tracker.close()
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

    dist.barrier(group=global_sync_group)
    dist.destroy_process_group(global_sync_group)
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
    parser.add_argument(
        "--eval_interval",
        type=int,
        default=50,
        help="Number of outer iterations between environment evaluations.",
    )

    parser.add_argument("--pipeline_stages", type=int, default=4)
    parser.add_argument("--data_parallel_groups", type=int, default=2)
    parser.add_argument(
        "--device_groups",
        type=str,
        default=None,
        help="Semicolon separated GPU ids for each pipeline group, e.g. '0,1,2,3;4,5,6,7'.",
    )

    parser.add_argument("--dist_backend", type=str, default="nccl")
    parser.add_argument("--resume_checkpoint", type=str, default=None, help="Path to checkpoint to resume from.")
    parser.add_argument(
        "--resume_iter",
        type=int,
        default=None,
        help="Completed outer iterations when resuming; overrides the value inferred from the checkpoint file name.",
    )
    parser.add_argument(
        "--resume_updates",
        type=int,
        default=None,
        help="Completed optimizer updates when resuming; overrides the value inferred from the checkpoint file name.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--log_stage_steps",
        action="store_true",
        help="Print per-microbatch pipeline transfer logs (disabled by default).",
    )
    parser.add_argument(
        "--progress_refresh",
        type=float,
        default=2.0,
        help="Seconds between progress bar refreshes on rank 0.",
    )

    args = parser.parse_args()

    if args.batch_size % args.micro_batches != 0:
        raise ValueError("batch_size must be divisible by micro_batches for pipeline parallelism.")

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for pipeline training.")

    train(args)


if __name__ == "__main__":
    main()
