"""
Quick test script to verify DDP setup is working correctly.

This script performs a minimal test of the DDP infrastructure without
running a full training session.

Usage:
    python test_ddp_setup.py [world_size]

Example:
    python test_ddp_setup.py 8
"""

import os
os.environ['MUJOCO_GL'] = 'egl'

import sys
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from termcolor import colored


def test_worker(rank, world_size):
    """Test worker function."""
    try:
        # Setup
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12355'

        dist.init_process_group(
            backend='nccl',
            rank=rank,
            world_size=world_size
        )

        torch.cuda.set_device(rank)
        device = f'cuda:{rank}'

        # Test 1: Basic tensor operations
        tensor = torch.ones(10, device=device) * rank
        if rank == 0:
            print(f'\n{colored("Test 1: Basic tensor operations", "cyan", attrs=["bold"])}')
        print(f'[Rank {rank}] Created tensor on {device}: mean={tensor.mean().item():.2f}')

        # Test 2: All-reduce operation
        dist.barrier()
        if rank == 0:
            print(f'\n{colored("Test 2: All-reduce operation", "cyan", attrs=["bold"])}')

        test_tensor = torch.tensor([float(rank)], device=device)
        dist.all_reduce(test_tensor, op=dist.ReduceOp.SUM)

        expected_sum = sum(range(world_size))
        assert test_tensor.item() == expected_sum, f'Expected {expected_sum}, got {test_tensor.item()}'
        print(f'[Rank {rank}] All-reduce successful: {test_tensor.item():.0f} (expected {expected_sum})')

        # Test 3: Broadcast operation
        dist.barrier()
        if rank == 0:
            print(f'\n{colored("Test 3: Broadcast operation", "cyan", attrs=["bold"])}')

        if rank == 0:
            broadcast_tensor = torch.tensor([42.0], device=device)
        else:
            broadcast_tensor = torch.tensor([0.0], device=device)

        dist.broadcast(broadcast_tensor, src=0)
        assert broadcast_tensor.item() == 42.0
        print(f'[Rank {rank}] Broadcast successful: {broadcast_tensor.item():.0f}')

        # Test 4: Gradient synchronization simulation
        dist.barrier()
        if rank == 0:
            print(f'\n{colored("Test 4: Gradient synchronization", "cyan", attrs=["bold"])}')

        model = torch.nn.Linear(10, 10).to(device)
        ddp_model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[rank],
            output_device=rank
        )

        # Simulate forward pass
        input_tensor = torch.randn(5, 10, device=device)
        output = ddp_model(input_tensor)
        loss = output.sum()
        loss.backward()

        print(f'[Rank {rank}] DDP model gradient sync successful')

        # Test 5: Memory availability
        dist.barrier()
        if rank == 0:
            print(f'\n{colored("Test 5: GPU memory", "cyan", attrs=["bold"])}')

        allocated = torch.cuda.memory_allocated(rank) / 1024**2  # MB
        reserved = torch.cuda.memory_reserved(rank) / 1024**2  # MB
        total = torch.cuda.get_device_properties(rank).total_memory / 1024**2  # MB

        print(f'[Rank {rank}] GPU {rank}: {allocated:.1f}MB allocated, '
              f'{reserved:.1f}MB reserved, {total:.1f}MB total')

        dist.barrier()

        if rank == 0:
            print(f'\n{colored("="*60, "green", attrs=["bold"])}')
            print(colored("All tests passed! DDP setup is working correctly.", "green", attrs=["bold"]))
            print(f'{colored("="*60, "green", attrs=["bold"])}\n')

        # Cleanup
        dist.destroy_process_group()

    except Exception as e:
        print(f'{colored(f"[Rank {rank}] Error: {e}", "red", attrs=["bold"])}')
        import traceback
        traceback.print_exc()
        raise


def main():
    """Main function."""
    world_size = int(sys.argv[1]) if len(sys.argv) > 1 else torch.cuda.device_count()

    if not torch.cuda.is_available():
        print(colored('Error: CUDA is not available', 'red', attrs=['bold']))
        sys.exit(1)

    num_gpus = torch.cuda.device_count()
    print(f'\n{colored("="*60, "cyan", attrs=["bold"])}')
    print(colored("DDP Setup Test", "cyan", attrs=["bold"]))
    print(f'{colored("="*60, "cyan", attrs=["bold"])}\n')
    print(f'Available GPUs: {num_gpus}')
    print(f'Testing with:   {world_size} GPUs')
    print(f'{colored("="*60, "cyan", attrs=["bold"])}\n')

    if world_size > num_gpus:
        print(colored(f'Error: Requested {world_size} GPUs but only {num_gpus} available', 'red', attrs=['bold']))
        sys.exit(1)

    if world_size < 2:
        print(colored('Warning: Testing with less than 2 GPUs. Some tests may be trivial.', 'yellow'))

    try:
        mp.spawn(
            test_worker,
            args=(world_size,),
            nprocs=world_size,
            join=True
        )
    except Exception as e:
        print(f'\n{colored("="*60, "red", attrs=["bold"])}')
        print(colored(f'Test failed: {e}', "red", attrs=["bold"]))
        print(f'{colored("="*60, "red", attrs=["bold"])}\n')
        sys.exit(1)


if __name__ == '__main__':
    main()
