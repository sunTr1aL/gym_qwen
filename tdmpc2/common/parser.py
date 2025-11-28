import copy
import dataclasses
import re
from pathlib import Path
from typing import Any

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf

from . import MODEL_SIZE, TASK_SET
from ..envs import make_env


def _get_base_dir_for_cfg() -> Path:
    """Return a base directory for cfg.work_dir that works with or without Hydra.

    When executed under a Hydra launcher, `hydra.utils.get_original_cwd()` points
    to the original working directory. Offline scripts that call `parse_cfg`
    directly (without initializing Hydra) should gracefully fall back to the
    current working directory instead of raising a ValueError.
    """

    try:
        return Path(hydra.utils.get_original_cwd())
    except Exception:
        return Path.cwd()


def cfg_to_dataclass(cfg, frozen: bool = False):
    """Convert an OmegaConf config to a dataclass object.

    This prevents graph breaks when used with torch.compile.
    """

    cfg_dict = OmegaConf.to_container(cfg)
    fields = []
    for key, value in cfg_dict.items():
        fields.append((key, Any, dataclasses.field(default_factory=lambda value_=value: value_)))
    dataclass_name = "Config"
    dataclass = dataclasses.make_dataclass(dataclass_name, fields, frozen=frozen)

    def get(self, val, default=None):
        return getattr(self, val, default)

    dataclass.get = get
    return dataclass()


def parse_cfg(cfg: OmegaConf) -> OmegaConf:
    """Parse a Hydra config. Mostly for convenience."""

    # Logic
    for k in cfg.keys():
        try:
            v = cfg[k]
            if v is None:
                v = True
        except Exception:
            pass

    # Algebraic expressions
    for k in cfg.keys():
        try:
            v = cfg[k]
            if isinstance(v, str):
                match = re.match(r"(\d+)([+\-*/])(\d+)", v)
                if match:
                    cfg[k] = eval(match.group(1) + match.group(2) + match.group(3))
                    if isinstance(cfg[k], float) and cfg[k].is_integer():
                        cfg[k] = int(cfg[k])
        except Exception:
            pass

    # Convenience
    base_dir = _get_base_dir_for_cfg()
    cfg.work_dir = base_dir / "logs" / cfg.task / str(cfg.seed) / cfg.exp_name
    cfg.task_title = cfg.task.replace("-", " ").title()
    cfg.bin_size = (cfg.vmax - cfg.vmin) / (cfg.num_bins - 1)  # Bin size for discrete regression
    cfg.device = str(cfg.get("device", "cuda"))

    # Optional overrides
    tasks_override = cfg.get("tasks_override")
    if tasks_override is not None and not isinstance(tasks_override, (list, tuple)):
        tasks_override = [tasks_override]
    cfg.tasks_override = tasks_override
    force_multitask = cfg.get("force_multitask", False)

    # Model size
    if cfg.get("model_size", None) is not None:
        assert cfg.model_size in MODEL_SIZE.keys(), (
            f"Invalid model size {cfg.model_size}. Must be one of {list(MODEL_SIZE.keys())}"
        )
        for k, v in MODEL_SIZE[cfg.model_size].items():
            cfg[k] = v
        if cfg.task == "mt30" and cfg.model_size == 19:
            cfg.latent_dim = 512  # This checkpoint is slightly smaller

    # Multi-task
    cfg.multitask = bool(force_multitask) or cfg.task in TASK_SET.keys()
    if tasks_override:
        cfg.tasks = list(tasks_override)
        cfg.multitask = True
    else:
        cfg.tasks = TASK_SET.get(cfg.task, [cfg.task])
    if cfg.multitask:
        if cfg.task in TASK_SET.keys():
            cfg.task_title = cfg.task.upper()
        # Account for slight inconsistency in task_dim for the mt30 experiments
        default_task_dim = 96 if cfg.task == "mt80" or cfg.get("model_size", 5) in {1, 317} else 64
        if not isinstance(cfg.get("task_dim", None), (int, float)):
            cfg.task_dim = default_task_dim
    else:
        cfg.task_dim = 0

    # Normalize action dimensions to integers for downstream tensor shapes.
    action_dim = cfg.get("action_dim", None)
    if action_dim is not None:
        try:
            cfg.action_dim = int(action_dim)
        except Exception:
            cfg.action_dim = action_dim

    action_dims = cfg.get("action_dims", None)
    if action_dims is not None:
        if not isinstance(action_dims, (list, tuple)):
            action_dims = [action_dims]
        try:
            cfg.action_dims = [int(d) for d in action_dims]
        except Exception:
            cfg.action_dims = action_dims

    return cfg_to_dataclass(cfg)


def populate_env_dims(cfg):
    """Ensure cfg has concrete obs_dim and action_dim based on an env.

    For multi-task configs (cfg.tasks is a list), we construct a temporary
    single-task cfg and build an env from that to infer dims.

    Returns:
        cfg, env_for_dims
    """

    pretrained_aligned = bool(getattr(cfg, "pretrained_aligned", False))

    ready_fields = (
        getattr(cfg, "obs_shape", None),
        getattr(cfg, "obs_dim", None),
        getattr(cfg, "action_dim", None),
        getattr(cfg, "action_dims", None),
        getattr(cfg, "episode_length", None),
    )
    if pretrained_aligned and all(val is not None and val != "???" for val in ready_fields):
        return cfg, None

    task = getattr(cfg, "task", None)
    tasks = getattr(cfg, "tasks", None)
    collection_mode = getattr(cfg, "collection_mode", "single")

    def _flatten_tasks_list(tasks_val):
        flat = []
        if tasks_val is not None:
            if isinstance(tasks_val, (list, tuple)):
                for t in tasks_val:
                    if isinstance(t, (list, tuple)):
                        flat.extend(t)
                    else:
                        flat.append(t)
            else:
                flat = [tasks_val]
        return [t for t in flat if isinstance(t, str)]

    flat_tasks = _flatten_tasks_list(tasks)

    single_task = None
    tasks_for_dims = flat_tasks if flat_tasks else []

    if collection_mode == "single":
        if isinstance(task, str):
            single_task = task
        elif flat_tasks:
            single_task = flat_tasks[0]
        if single_task is None:
            raise ValueError("populate_env_dims: neither cfg.task nor cfg.tasks provide a string task")
        tasks_for_dims = [single_task]
    else:
        if flat_tasks:
            single_task = flat_tasks[0]
        elif isinstance(task, str):
            single_task = task
            tasks_for_dims = [task]
        else:
            raise ValueError("populate_env_dims: neither cfg.task nor cfg.tasks provide a string task")

    cfg_env = copy.deepcopy(cfg)
    cfg_env.task = single_task

    if hasattr(cfg_env, "tasks"):
        cfg_env.tasks = None
    if hasattr(cfg_env, "multitask"):
        cfg_env.multitask = False
    if hasattr(cfg_env, "multi_task"):
        cfg_env.multi_task = False
    if hasattr(cfg_env, "num_tasks"):
        cfg_env.num_tasks = 1
    if hasattr(cfg_env, "collection_mode"):
        cfg_env.collection_mode = "single"

    env = make_env(cfg_env)

    obs_type = getattr(cfg, "obs_type", None)
    if not isinstance(obs_type, str) or obs_type == "???":
        obs_type = "states"
        cfg.obs_type = obs_type
        cfg.obs = obs_type

    if (
        not hasattr(cfg, "obs_shape")
        or isinstance(cfg.obs_shape, str)
        or getattr(cfg, "obs_shape", None) == "???"
    ):
        if hasattr(env, "obs_shape"):
            obs_shape_env = env.obs_shape
            if isinstance(obs_shape_env, dict):
                cfg.obs_shape = obs_shape_env
            else:
                cfg.obs_shape = {obs_type: tuple(obs_shape_env)}
        else:
            cfg.obs_shape = {obs_type: tuple(env.observation_space.shape)}

    if (
        not hasattr(cfg, "obs_dim")
        or isinstance(cfg.obs_dim, str)
        or getattr(cfg, "obs_dim", None) == "???"
    ):
        obs_shape_cfg = getattr(cfg, "obs_shape", None)
        if isinstance(obs_shape_cfg, dict):
            if obs_type in obs_shape_cfg:
                shape_for_dim = obs_shape_cfg[obs_type]
            else:
                first_key = next(iter(obs_shape_cfg.keys()))
                shape_for_dim = obs_shape_cfg[first_key]
            cfg.obs_dim = int(shape_for_dim[0])
        elif hasattr(env, "obs_shape"):
            if isinstance(env.obs_shape, dict):
                first_key = next(iter(env.obs_shape.keys()))
                cfg.obs_dim = int(env.obs_shape[first_key][0])
            else:
                cfg.obs_dim = int(env.obs_shape[0])
        else:
            cfg.obs_dim = int(env.observation_space.shape[0])

    if (
        not hasattr(cfg, "obs_shape")
        or isinstance(cfg.obs_shape, str)
        or getattr(cfg, "obs_shape", None) == "???"
    ):
        if hasattr(env, "obs_shape"):
            obs_shape_env = env.obs_shape
            if isinstance(obs_shape_env, dict):
                cfg.obs_shape = obs_shape_env
            else:
                cfg.obs_shape = {"obs": tuple(obs_shape_env)}
        else:
            cfg.obs_shape = {"obs": tuple(env.observation_space.shape)}

    if (
        not hasattr(cfg, "action_dim")
        or isinstance(cfg.action_dim, str)
        or getattr(cfg, "action_dim", None) == "???"
    ):
        if hasattr(env, "action_dim"):
            cfg.action_dim = int(env.action_dim)
        else:
            cfg.action_dim = int(env.action_space.shape[0])

    num_tasks_for_dims = max(1, len(tasks_for_dims))

    cfg.action_dim = int(cfg.action_dim)

    if (
        not hasattr(cfg, "action_dims")
        or isinstance(cfg.action_dims, str)
        or getattr(cfg, "action_dims", None) == "???"
    ):
        cfg.action_dims = [cfg.action_dim] * num_tasks_for_dims

    if hasattr(cfg_env, "episode_length"):
        cfg.episode_length = cfg_env.episode_length

    return cfg, env
