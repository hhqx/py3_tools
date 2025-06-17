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

  # Or manually specify which ranks should fail:
  torchrun --nnodes=1 --nproc_per_node=3 debug_multi_torch_rank.py --fail_ranks 0 1 --debug

  # Specify debug mode (console, web, socket):
  export IPDB_MODE=socket
  torchrun --nnodes=1 --nproc_per_node=3 debug_multi_torch_rank.py --fail_ranks 0,2
  
  # In another terminal when using socket mode:
  # nc -U /tmp/pdb.sock.2
  
  # Set logging level:
  torchrun --nnodes=1 --nproc_per_node=2 debug_multi_torch_rank.py --fail_ranks 1 --log_level DEBUG
  
  # Specify custom timeout for distributed operations:
  torchrun --nnodes=1 --nproc_per_node=2 debug_multi_torch_rank.py --timeout 1800
"""

import os
import sys
import argparse
import logging
import datetime  # Add this import for timedelta
import torch
import torch.distributed as dist
from py3_tools.py_debug.debug_utils import Debugger, setup_logging

# Get module logger
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="PyTorch Distributed Debugging Example")
    parser.add_argument('--fail_ranks', type=int, nargs='+', default=[1],
                        help='Which ranks should throw an error')
    parser.add_argument('--debug', action='store_true', 
                        help='Enable debugging')
    parser.add_argument('--backend', type=str, default=None,
                        help='PyTorch distributed backend (nccl or gloo)')
    parser.add_argument('--debug_mode', choices=['console', 'web', 'socket'], 
                        help='Debug mode (overrides environment variable)')
    parser.add_argument('--error_type', choices=['indexerror', 'zerodivision', 'runtime'],
                        default='indexerror', help='Type of error to trigger')
    parser.add_argument('--log_level', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                       default='INFO', help='Set logging level')
    parser.add_argument('--timeout', type=int, default=600,
                       help='Timeout in seconds for distributed operations (default: 10 minutes)')
    args = parser.parse_args()

    # Initialize the distributed environment (torchrun should have set the env variables)
    if not args.backend:
        args.backend = 'nccl' if torch.cuda.is_available() else 'gloo'
    
    try:
        init_process_group(args.backend, timeout=args.timeout)
        # Get rank for logging
        rank = dist.get_rank()
    except Exception as e:
        logger.error(f"Failed to initialize process group: {e}")
        logger.error("Make sure to run this script with torchrun")
        sys.exit(1)
        
    # Set up logging with rank information
    log_level = getattr(logging, args.log_level)
    setup_logging(level=log_level, rank=rank)
    
    # Enable debug if requested via command line
    if args.debug:
        Debugger.debug_flag = True
        logger.info("Debugging enabled via command line flag")
    elif Debugger.debug_flag:
        logger.info("Debugging enabled via environment variable")
    
    # Set debug mode if specified
    if args.debug_mode:
        Debugger.debug_mode = args.debug_mode
        logger.info(f"Debug mode set to: {args.debug_mode}")

    # Get distributed info
    world_size = dist.get_world_size()

    logger.info(f"Initialized process group: {args.backend}, world_size={world_size}, timeout={args.timeout}s")

    # Example functions with debugging
    @Debugger.on_error()
    def process_tensor():
        """Creates a tensor and performs operations based on rank."""
        # Each rank creates a tensor
        tensor = torch.ones(10) * rank
        
        # All ranks log their tensor
        logger.debug(f"Created tensor: {tensor}")
        
        # Synchronize all processes
        logger.debug("Synchronizing processes with barrier()")
        dist.barrier()
        
        # The specified ranks will throw an error
        if rank in args.fail_ranks:
            logger.warning(f"About to cause an error of type '{args.error_type}'...")
            
            if args.error_type == 'indexerror':
                # Generate an out-of-bounds error
                logger.debug("Attempting to access index 20 (out of bounds)")
                bad_value = tensor[20]  # This will raise an IndexError
            elif args.error_type == 'zerodivision':
                # Generate a division by zero error
                logger.debug("Attempting division by zero")
                result = tensor[0] / 0
            else:
                # Generate a runtime error
                logger.debug("Raising RuntimeError")
                raise RuntimeError(f"Simulated error on rank {rank}")
            
        # Perform a collective operation
        logger.debug("Performing all_reduce operation")
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        logger.info(f"After all_reduce: {tensor}")
        
        return tensor

    logger.debug("Starting first tensor operation")
    try:
        result = process_tensor()
        logger.info("First tensor operation completed successfully")
    except Exception as e:
        logger.error(f"Error during first tensor operation: {e}")
        raise
    
    logger.debug("Starting second tensor operation")
    try:
        # Run the function that will error on the specified rank
        result = process_tensor()
        logger.info("Process completed successfully")
    except Exception as e:
        logger.error(f"Error caught: {e}")
        raise
    # finally:
    #     # Clean up
    #     logger.debug("Destroying process group")
    #     dist.destroy_process_group()
    #     logger.info("Process group destroyed")

def init_process_group(backend, timeout=600):
    """Initialize the distributed process group.
    
    Args:
        backend (str): PyTorch distributed backend ('nccl' or 'gloo')
        timeout (int): Timeout in seconds for distributed operations
    """
    # Convert timeout to timedelta
    timeout_delta = datetime.timedelta(seconds=timeout)
    
    # Check if we're using torchrun with the required env vars set
    required_vars = ['RANK', 'WORLD_SIZE', 'MASTER_ADDR', 'MASTER_PORT']
    if all(var in os.environ for var in required_vars):
        logger.debug(f"Found distributed environment variables, initializing with {backend} backend (timeout: {timeout}s)")
        dist.init_process_group(backend=backend, timeout=timeout_delta)
    else:
        # Fallback to a local setup
        logger.warning("Missing distributed environment variables, falling back to local setup")
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '29500'
        if 'RANK' not in os.environ:
            os.environ['RANK'] = '0'
        if 'WORLD_SIZE' not in os.environ:
            os.environ['WORLD_SIZE'] = '1'
        logger.debug(f"Using local setup with RANK={os.environ['RANK']}, WORLD_SIZE={os.environ['WORLD_SIZE']}, timeout={timeout}s")
        dist.init_process_group(backend=backend, timeout=timeout_delta)

if __name__ == "__main__":
    main()
