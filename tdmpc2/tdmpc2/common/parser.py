import dataclasses
import re
from pathlib import Path
from typing import Any

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf

from tdmpc2.common import MODEL_SIZE, TASK_SET
from tdmpc2.envs import make_env


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


def cfg_to_dataclass(cfg, frozen=False):
	"""
	Converts an OmegaConf config to a dataclass object.
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
        """
        Parses a Hydra config. Mostly for convenience.
        """

	# Logic
	for k in cfg.keys():
		try:
			v = cfg[k]
			if v == None:
				v = True
		except:
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
		except:
			pass

	# Convenience
	base_dir = _get_base_dir_for_cfg()
	cfg.work_dir = base_dir / 'logs' / cfg.task / str(cfg.seed) / cfg.exp_name
	cfg.task_title = cfg.task.replace("-", " ").title()
	cfg.bin_size = (cfg.vmax - cfg.vmin) / (cfg.num_bins-1) # Bin size for discrete regression
	cfg.device = str(cfg.get('device', 'cuda'))

	# Optional overrides
	tasks_override = cfg.get('tasks_override')
	if tasks_override is not None and not isinstance(tasks_override, (list, tuple)):
		tasks_override = [tasks_override]
	cfg.tasks_override = tasks_override
	force_multitask = cfg.get('force_multitask', False)

	# Model size
	if cfg.get('model_size', None) is not None:
		assert cfg.model_size in MODEL_SIZE.keys(), \
			f'Invalid model size {cfg.model_size}. Must be one of {list(MODEL_SIZE.keys())}'
		for k, v in MODEL_SIZE[cfg.model_size].items():
			cfg[k] = v
		if cfg.task == 'mt30' and cfg.model_size == 19:
			cfg.latent_dim = 512 # This checkpoint is slightly smaller

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
		default_task_dim = 96 if cfg.task == 'mt80' or cfg.get('model_size', 5) in {1, 317} else 64
		if not isinstance(cfg.get('task_dim', None), (int, float)):
			cfg.task_dim = default_task_dim
	else:
		cfg.task_dim = 0

	# Normalize action dimensions to integers for downstream tensor shapes.
	action_dim = cfg.get('action_dim', None)
	if action_dim is not None:
		try:
			cfg.action_dim = int(action_dim)
		except Exception:
			cfg.action_dim = action_dim

        action_dims = cfg.get('action_dims', None)
        if action_dims is not None:
                if not isinstance(action_dims, (list, tuple)):
                        action_dims = [action_dims]
                try:
                        cfg.action_dims = [int(d) for d in action_dims]
                except Exception:
                        cfg.action_dims = action_dims

        return cfg_to_dataclass(cfg)


def populate_env_dims(cfg):
        """
        Ensure cfg has concrete obs_dim and action_dim based on an environment
        created with make_env. This must work both when called from Hydra
        training code and from offline scripts (like collect_corrector_data).

        - If cfg.tasks is a list/tuple, we use the first task as cfg.task.
        - If cfg.task is already set, we use it.
        - If obs_dim/action_dim are strings or "???", we overwrite them using
          the env's shapes.
        Returns:
            cfg, env
        """
        # Resolve a single task string
        task = getattr(cfg, "task", None)
        tasks = getattr(cfg, "tasks", None)
        if task is None and tasks is not None:
                # use the first task in the list/tuple
                if isinstance(tasks, (list, tuple)) and len(tasks) > 0:
                        task = tasks[0]
                        cfg.task = task

        if task is None:
                raise ValueError("populate_env_dims: cfg.task is not set and cfg.tasks is empty")

        # Build env similarly to training code.
        # Use sensible defaults for optional fields if they are missing.
        obs_type     = getattr(cfg, "obs_type", "states")
        frame_stack  = int(getattr(cfg, "frame_stack", 1))
        action_repeat = int(getattr(cfg, "action_repeat", 1))
        seed         = int(getattr(cfg, "seed", 0))

        env = make_env(
                task=task,
                obs_type=obs_type,
                frame_stack=frame_stack,
                action_repeat=action_repeat,
                seed=seed,
        )

        # Infer obs_dim
        if (not hasattr(cfg, "obs_dim") or
            isinstance(cfg.obs_dim, str) or
            getattr(cfg, "obs_dim", None) == "???"):
                if hasattr(env, "obs_shape"):
                        cfg.obs_dim = int(env.obs_shape[0])
                else:
                        # fallback to standard gym space
                        cfg.obs_dim = int(env.observation_space.shape[0])

        # Infer action_dim
        if (not hasattr(cfg, "action_dim") or
            isinstance(cfg.action_dim, str) or
            getattr(cfg, "action_dim", None) == "???"):
                if hasattr(env, "action_dim"):
                        cfg.action_dim = int(env.action_dim)
                else:
                        cfg.action_dim = int(env.action_space.shape[0])

        # You can add similar logic here for other dims (proprio_dim, etc.) if needed.

        return cfg, env
