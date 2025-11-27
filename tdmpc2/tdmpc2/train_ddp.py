"""
Distributed training script for TD-MPC2 with online training support.

Supports multi-GPU training using PyTorch DDP (DistributedDataParallel).

Example usage:
    # Single task with 8 GPUs
    python train_ddp.py task=dog-run world_size=8 model_size=5

    # Custom sync frequency (sync gradients every 2 updates)
    python train_ddp.py task=humanoid-run world_size=8 sync_freq=2

    # With specific port
    python train_ddp.py task=walker-run world_size=4 master_port=12355
"""

import os
os.environ['MUJOCO_GL'] = os.getenv("MUJOCO_GL", 'egl')
os.environ['LAZY_LEGACY_OP'] = '0'
os.environ['TORCHDYNAMO_INLINE_INBUILT_NN_MODULES'] = "1"
os.environ['TORCH_LOGS'] = "+recompiles"

import warnings
warnings.filterwarnings('ignore')
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

import hydra
from termcolor import colored

from tdmpc2.common.parser import parse_cfg
from tdmpc2.common.seed import set_seed
from tdmpc2.common.buffer import Buffer
from tdmpc2.envs import make_env
from tdmpc2.tdmpc2_ddp import TDMPC2_DDP
from tdmpc2.trainer.ddp_online_trainer import DDPOnlineTrainer
from tdmpc2.common.logger import Logger

torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision('high')


def setup_ddp(rank, world_size, master_addr='localhost', master_port='12355'):
	"""
	Initialize the distributed environment.

	Args:
		rank: Unique identifier of each process
		world_size: Total number of processes
		master_addr: Address of the master node
		master_port: Port for communication
	"""
	os.environ['MASTER_ADDR'] = master_addr
	os.environ['MASTER_PORT'] = master_port

	# Initialize the process group
	dist.init_process_group(
		backend='nccl',  # Use NCCL for GPU communication
		init_method='env://',
		rank=rank,
		world_size=world_size
	)

	# Set the device for this process
	torch.cuda.set_device(rank)

	if rank == 0:
		print(f'Initialized DDP: world_size={world_size}, backend=nccl')


def cleanup_ddp():
	"""Clean up the distributed environment."""
	dist.destroy_process_group()


def train_worker(rank, cfg):
	"""
	Training worker function for each process.

	Args:
		rank: Process rank (GPU ID)
		cfg: Configuration dictionary
	"""
	# Setup distributed training
	master_addr = getattr(cfg, 'master_addr', 'localhost')
	master_port = getattr(cfg, 'master_port', '12355')
	setup_ddp(rank, cfg.world_size, master_addr, master_port)

	# Set device
	device = f'cuda:{rank}'

	# Set seed (different for each rank to ensure diversity)
	set_seed(cfg.seed + rank)

	# Add rank and device to config
	cfg.rank = rank
	cfg.device = device

	if rank == 0:
		print(colored('Work dir:', 'yellow', attrs=['bold']), cfg.work_dir)
		print(colored('Device:', 'yellow', attrs=['bold']), device)
		print(colored('World size:', 'yellow', attrs=['bold']), cfg.world_size)
		print(colored('Sync frequency:', 'yellow', attrs=['bold']), getattr(cfg, 'sync_freq', 1))

        try:
                # Create environment (each process has its own environment)
                env = make_env(cfg)

                # Ensure all ranks share identical environment-derived config
                env_cfg = {
                        "obs_shape": getattr(cfg, "obs_shape", None),
                        "obs_shapes": getattr(cfg, "obs_shapes", None),
                        "action_dim": getattr(cfg, "action_dim", None),
                        "action_dims": getattr(cfg, "action_dims", None),
                        "episode_length": getattr(cfg, "episode_length", None),
                        "episode_lengths": getattr(cfg, "episode_lengths", None),
                        "seed_steps": getattr(cfg, "seed_steps", None),
                }
                env_cfg_list = [env_cfg]
                dist.broadcast_object_list(env_cfg_list, src=0)
                for k, v in env_cfg_list[0].items():
                        setattr(cfg, k, v)

                # Create agent with DDP wrapper
                agent = TDMPC2_DDP(cfg)

		# Create buffer (each process has its own buffer)
		buffer = Buffer(cfg)

		# Create logger (only rank 0 logs)
		logger = Logger(cfg) if rank == 0 else None

		# Create trainer
		trainer = DDPOnlineTrainer(
			cfg=cfg,
			env=env,
			agent=agent,
			buffer=buffer,
			logger=logger if logger else Logger(cfg),  # Provide dummy logger for non-zero ranks
		)

		# Train
		trainer.train()

		if rank == 0:
			print('\nTraining completed successfully')

	except Exception as e:
		print(f'Rank {rank} encountered error: {e}')
		import traceback
		traceback.print_exc()
		raise

	finally:
		# Cleanup
		cleanup_ddp()


@hydra.main(config_name='config', config_path='.')
def main(cfg: dict):
	"""
	Main function to launch distributed training.

	Args:
		cfg: Hydra configuration dictionary
	"""
	# Validate configuration
	assert torch.cuda.is_available(), 'CUDA is not available'

	# Set default world_size if not specified
	if not hasattr(cfg, 'world_size'):
		cfg.world_size = torch.cuda.device_count()
		print(f'world_size not specified, using all available GPUs: {cfg.world_size}')

	assert cfg.world_size > 0, 'world_size must be greater than 0'
	assert cfg.world_size <= torch.cuda.device_count(), \
		f'world_size ({cfg.world_size}) cannot exceed number of available GPUs ({torch.cuda.device_count()})'

	assert cfg.steps > 0, 'Must train for at least 1 step.'

	# Parse configuration
	cfg = parse_cfg(cfg)

	# Set default sync frequency
	if not hasattr(cfg, 'sync_freq'):
		cfg.sync_freq = 1  # Sync every update by default

	print(colored('\n' + '='*60, 'cyan', attrs=['bold']))
	print(colored('TD-MPC2 Distributed Training', 'cyan', attrs=['bold']))
	print(colored('='*60 + '\n', 'cyan', attrs=['bold']))
	print(colored(f'Task: {cfg.task}', 'green'))
	print(colored(f'GPUs: {cfg.world_size}', 'green'))
	print(colored(f'Steps: {cfg.steps}', 'green'))
	print(colored(f'Batch size (per GPU): {cfg.batch_size}', 'green'))
	print(colored(f'Effective batch size: {cfg.batch_size * cfg.world_size}', 'green', attrs=['bold']))
	print(colored(f'Gradient sync frequency: {cfg.sync_freq}', 'green'))
	print(colored('\n' + '='*60 + '\n', 'cyan', attrs=['bold']))

	# Launch distributed training
	if cfg.world_size == 1:
		# Single GPU training (no need for distributed)
		print(colored('Warning: world_size=1, running in single-GPU mode', 'yellow'))
		train_worker(0, cfg)
	else:
		# Multi-GPU training
		mp.spawn(
			train_worker,
			args=(cfg,),
			nprocs=cfg.world_size,
			join=True
		)


if __name__ == '__main__':
	main()
