#!/usr/bin/env python
"""
Multi-Process Distributed Debugging Example

This script demonstrates how to use the Debugger utility in a multi-process
PyTorch distributed environment. It shows how different ranks are handled
during debugging.

Basic Usage:
  # Run with 2 processes:
  export IPDB_DEBUG=1
  torchrun --nnodes=1 --nproc_per_node=2 debug_multi_torch_rank.py

  # Or manually specify which ranks should fail:
  torchrun --nnodes=1 --nproc_per_node=3 debug_multi_torch_rank.py --fail_ranks 0 1 --debug

Breakpoint Usage Examples:

  1. Debug specific ranks at tensor creation:
     export IPDB_DEBUG=1
     torchrun --nnodes=1 --nproc_per_node=3 debug_multi_torch_rank.py \
       --breakpoint_ranks 0 2 --breakpoint_mode after_tensor

  2. Debug before all_reduce operation:
     export IPDB_DEBUG=1
     torchrun --nnodes=1 --nproc_per_node=4 debug_multi_torch_rank.py \
       --breakpoint_ranks 1 3 --breakpoint_mode before_allreduce

  3. Debug only the rank that will fail:
     export IPDB_DEBUG=1
     torchrun --nnodes=1 --nproc_per_node=3 debug_multi_torch_rank.py \
       --fail_ranks 2 --breakpoint_mode before_error

  4. Step-by-step debugging with multiple breakpoints:
     export IPDB_DEBUG=1
     torchrun --nnodes=1 --nproc_per_node=2 debug_multi_torch_rank.py \
       --step_debug --breakpoint_ranks 0 1

  5. Debug all stages with specific ranks:
     export IPDB_DEBUG=1
     torchrun --nnodes=1 --nproc_per_node=3 debug_multi_torch_rank.py \
       --breakpoint_ranks 0 2 --breakpoint_mode all

Socket-based Debugging:
  # Specify debug mode (console, web, socket):
  export IPDB_MODE=socket
  torchrun --nnodes=1 --nproc_per_node=3 debug_multi_torch_rank.py \
    --fail_ranks 0,2 --breakpoint_ranks 1 --breakpoint_mode before_error
  
  # Connect to the socket debugger:
  # socat $(tty),raw,echo=0 UNIX-CONNECT:/tmp/pdb.sock.1
  
  # For better terminal support with arrow keys:
  # Use stty first to set your terminal to raw mode:
  #   stty raw -echo
  #   socat $(tty),raw,echo=0 UNIX-CONNECT:/tmp/pdb.sock.1
  #   # When done: stty sane
  # 
  # Or use rlwrap for readline support (recommended):
  #   rlwrap socat $(tty),raw,echo=0 UNIX-CONNECT:/tmp/pdb.sock.1

Programmatic Breakpoint Usage in Code:

  from py3_tools.py_debug import breakpoint

  # Debug all ranks at current location
  breakpoint()

  # Debug specific ranks only
  breakpoint(ranks=[0, 2])

  # Debug single rank
  breakpoint(ranks=1)

  # Conditional debugging based on rank
  if dist.get_rank() in [0, 1]:
      breakpoint(ranks=[dist.get_rank()])

Advanced Examples:

  # Debug with custom socket configuration:
  export IPDB_MODE=socket
  export IPDB_CONNECTION_TYPE=tcp
  export IPDB_HOST=0.0.0.0
  export IPDB_PORT=5678
  torchrun --nnodes=1 --nproc_per_node=2 debug_multi_torch_rank.py \
    --breakpoint_ranks 1 --debug

  # Then connect with: socat $(tty),raw,echo=0 TCP:localhost:5679  # port = 5678 + rank

  # Debug with web interface:
  export IPDB_MODE=web
  torchrun --nnodes=1 --nproc_per_node=2 debug_multi_torch_rank.py \
    --breakpoint_ranks 1 --debug
  # Open http://localhost:4445 in browser (port = 4444 + rank)

Command-line Arguments:
  --fail_ranks RANK [RANK ...]     Which ranks should throw an error (default: [1])
  --breakpoint_ranks RANK [RANK ...] Which ranks should trigger breakpoints
  --breakpoint_mode {before_error,after_tensor,before_allreduce,all}
                                   When to trigger breakpoints (default: before_error)
  --step_debug                     Enable step-by-step debugging with multiple breakpoints
  --debug                          Enable debugging manually
  --debug_mode {console,web,socket} Debug mode (overrides environment variable)
  --error_type {indexerror,zerodivision,runtime}
                                   Type of error to trigger (default: indexerror)
  --backend BACKEND                PyTorch distributed backend (nccl or gloo)
  --log_level {DEBUG,INFO,WARNING,ERROR,CRITICAL}
                                   Set logging level (default: INFO)
  --timeout SECONDS                Timeout for distributed operations (default: 600)

Socket Debugging Commands:
  # - h or help: Show help
  # - n or next: Execute next line (step over)
  # - s or step: Step into function
  # - c or continue: Continue execution
  # - q or quit: Quit debugger
  # - l or list: Show current line in context
  # - w or where: Show call stack
  # - p expression: Print value of expression
  # - pp expression: Pretty-print expression
  # - !command: Execute Python command
"""

import os
import sys
import argparse
import logging
import datetime  # Add this import for timedelta
import torch
import torch.distributed as dist
from py3_tools.py_debug.debug_utils import setup_logging
from py3_tools.py_debug import Debugger, breakpoint


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
    
    # Add breakpoint-related arguments
    parser.add_argument('--breakpoint_ranks', type=int, nargs='+', default=None,
                       help='Which ranks should trigger breakpoints (e.g., --breakpoint_ranks 0 1)')
    parser.add_argument('--breakpoint_mode', choices=['before_error', 'after_tensor', 'before_allreduce', 'all'], 
                       default='before_error', help='When to trigger breakpoints')
    parser.add_argument('--step_debug', action='store_true',
                       help='Enable step-by-step debugging with multiple breakpoints')
    
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
    @Debugger.attach_on_error()
    def process_tensor():
        """Creates a tensor and performs operations based on rank."""
        # Each rank creates a tensor
        tensor = torch.ones(10) * rank
        
        # All ranks log their tensor
        logger.debug(f"Created tensor: {tensor}")
        
        # Breakpoint example 1: Debug specific ranks after tensor creation
        if args.breakpoint_mode in ['after_tensor', 'all'] and args.breakpoint_ranks:
            logger.info(f"Setting breakpoint after tensor creation for ranks: {args.breakpoint_ranks}")
            breakpoint(ranks=args.breakpoint_ranks)
            
            a = 1
            
            breakpoint(ranks=args.breakpoint_ranks)
            
            b = 2
        
        # Synchronize all processes
        logger.debug("Synchronizing processes with barrier()")
        dist.barrier()
        
        # Breakpoint example 2: Debug before all_reduce operation
        if args.breakpoint_mode in ['before_allreduce', 'all'] and args.breakpoint_ranks:
            logger.info(f"Setting breakpoint before all_reduce for ranks: {args.breakpoint_ranks}")
            breakpoint(ranks=args.breakpoint_ranks)
        
        # Perform a collective operation
        logger.debug("Performing all_reduce operation")
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        logger.info(f"After all_reduce: {tensor}")
        
        # Breakpoint example 3: Debug before error (if this rank will fail)
        if args.breakpoint_mode in ['before_error', 'all'] and rank in args.fail_ranks:
            logger.warning(f"Setting breakpoint before error on rank {rank}")
            breakpoint(ranks=[rank])  # Only debug the rank that will fail
        
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
        
        return tensor

    # Step-by-step debugging example
    def step_debug_example():
        """Demonstrates step-by-step debugging across multiple operations."""
        logger.info("Starting step-by-step debugging example")
        
        # Step 1: Initial tensor creation
        logger.debug("Step 1: Creating initial tensor")
        x = torch.randn(5) + rank
        breakpoint(ranks=args.breakpoint_ranks or [0, 1])
        
        # Step 2: Tensor transformation
        logger.debug("Step 2: Transforming tensor")
        y = x * 2 + 1
        if args.step_debug:
            breakpoint(ranks=args.breakpoint_ranks or [rank])
        
        # Step 3: Distributed operation
        logger.debug("Step 3: All-gather operation")
        gathered = [torch.zeros_like(y) for _ in range(world_size)]
        dist.all_gather(gathered, y)
        if args.step_debug:
            breakpoint(ranks=args.breakpoint_ranks or [0])
        
        logger.info(f"Step debug completed on rank {rank}")
        return gathered

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
    
    # Run step debugging if requested
    if args.step_debug:
        logger.info("Running step-by-step debugging example")
        try:
            step_result = step_debug_example()
            logger.info("Step debugging completed successfully")
        except Exception as e:
            logger.error(f"Error during step debugging: {e}")
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
