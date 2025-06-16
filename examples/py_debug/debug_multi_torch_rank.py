#!/usr/bin/env python
"""
Multi-Process Distributed Debugging Example

This script demonstrates how to use the Debugger utility in a multi-process
PyTorch distributed environment. It shows how different ranks are handled
during debugging.

Usage:
  # Run with 2 processes:
  export IPDB_DEBUG=1
  torchrun --nnodes=1 --nproc_per_node=2 debug_multi_torch_rank.py

  # Or manually specify which rank should fail:
  torchrun --nnodes=1 --nproc_per_node=2 debug_multi_torch_rank.py --fail-rank 1 --debug
"""

import os
import sys
import argparse
import torch
import torch.distributed as dist
from py_tools.py_debug.debug_utils import Debugger

def main():
    parser = argparse.ArgumentParser(description="PyTorch Distributed Debugging Example")
    parser.add_argument('--fail-rank', type=int, default=1, 
                        help='Which rank should throw an error')
    parser.add_argument('--debug', action='store_true', 
                        help='Enable debugging')
    parser.add_argument('--backend', type=str, default=None,
                        help='PyTorch distributed backend (nccl or gloo)')
    args = parser.parse_args()

    # Enable debug if requested via command line
    if args.debug:
        Debugger.debug_flag = True

    # Initialize the distributed environment (torchrun should have set the env variables)
    if not args.backend:
        args.backend = 'nccl' if torch.cuda.is_available() else 'gloo'
    
    try:
        init_process_group(args.backend)
    except Exception as e:
        print(f"Failed to initialize process group: {e}")
        print("Make sure to run this script with torchrun")
        sys.exit(1)
    
    # Get distributed info
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    print(f"[Rank {rank}] Initialized process group: {args.backend}, "
          f"world_size={world_size}")

    # Example functions with debugging
    @Debugger.on_error()
    def process_tensor():
        """Creates a tensor and performs operations based on rank."""
        # Each rank creates a tensor
        tensor = torch.ones(10) * rank
        
        # All ranks print their tensor
        print(f"[Rank {rank}] Created tensor: {tensor}")
        
        # Synchronize all processes
        dist.barrier()
        
        # The specified rank will throw an error
        if 1 or rank == args.fail_rank:
            print(f"[Rank {rank}] About to cause an error...")
            # Generate an out-of-bounds error
            bad_value = tensor[20]  # This will raise an IndexError
            
        # Perform a collective operation
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        print(f"[Rank {rank}] After all_reduce: {tensor}")
        
        return tensor

    # result = process_tensor()
    # print(f"[Rank {rank}] Process completed successfully")
        
    try:
        # Run the function that will error on the specified rank
        result = process_tensor()
        print(f"[Rank {rank}] Process completed successfully")
    except Exception as e:
        print(f"[Rank {rank}] Error caught: {e}")
    finally:
        # Clean up
        dist.destroy_process_group()
        print(f"[Rank {rank}] Process group destroyed")

def init_process_group(backend):
    """Initialize the distributed process group."""
    # Check if we're using torchrun with the required env vars set
    required_vars = ['RANK', 'WORLD_SIZE', 'MASTER_ADDR', 'MASTER_PORT']
    if all(var in os.environ for var in required_vars):
        dist.init_process_group(backend=backend)
    else:
        # Fallback to a local setup
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '29500'
        if 'RANK' not in os.environ:
            os.environ['RANK'] = '0'
        if 'WORLD_SIZE' not in os.environ:
            os.environ['WORLD_SIZE'] = '1'
        dist.init_process_group(backend=backend)

if __name__ == "__main__":
    main()
