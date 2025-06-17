'''
debug_utils.py

A class-based utility for debugging functions using WebPdb, including torch.distributed multi-rank support.

Features:
- Debugger class: encapsulates configuration (remote web debugger base port, debug flag).
- @Debugger.on_error decorator: wraps function calls for exception-debugging based on rank.
- Supports 'nccl' backend for GPUs and 'gloo' backend for CPU-only distributed runs.
- **Rendezvous backend ('c10d')**: implements the rendezvous protocol using PyTorch's C++ c10d library.
- **Communication backends**:
  - **NCCL**: high-performance GPU collectives on NVIDIA hardware.
  - **Gloo**: CPU and GPU-capable fallback backend.
- **Socket-based debugging**: supports remote debugging through Unix sockets.

Usage Examples:

1. Single-process auto-debug via environment:

    export IPDB_DEBUG=1
    python debug_utils.py --mode error

2. Single-process manual debug via flag:

    python debug_utils.py --mode error --debug

3. Torchrun multi-process (2 ranks):

    export IPDB_DEBUG=1
    torchrun \
        --nnodes 1 \
        --nproc_per_node 2 \
        --rdzv_backend c10d \
        --rdzv_endpoint localhost:29500 \
        debug_utils.py --mode distributed_error

4. Socket-based debugging:
    
    export IPDB_DEBUG=1
    export IPDB_MODE=socket
    torchrun --nnodes=1 --nproc_per_node=2 debug_multi_torch_rank.py --fail_ranks 1
    # In another terminal: socat - UNIX-CONNECT:/tmp/pdb.sock.1

Command-line Options:
    --mode [hello|error|distributed_error]
    --debug        Manually enable debug regardless of environment
    --name         Name for hello
    --log_level    Set logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)

'''
import os
import argparse
import traceback
import ipdb
import sys
import pdb
import logging
import socket

# Distributed support
try:
    import torch
    import torch.distributed as dist
    _dist_available = True
except ImportError:
    _dist_available = False

# WebPdb support
try:
    from web_pdb import set_trace as web_set_trace
    from web_pdb import WebPdb
    _web_pdb_available = True
except ImportError:
    _web_pdb_available = False

# Configure module-level logger
logger = logging.getLogger(__name__)

# Socket-based debugging configuration
SOCK_PATH = '/tmp/pdb.sock'

def setup_logging(level=logging.INFO, rank=None):
    """Configure logging with appropriate format and level.
    
    Args:
        level (int): Logging level (e.g., logging.DEBUG, logging.INFO)
        rank (int, optional): Process rank for distributed environments
    """
    # Determine rank suffix for the format
    rank_suffix = f"[rank:{rank}] " if rank is not None else ""
    
    # Configure root logger
    root_logger = logging.getLogger()
    if not root_logger.handlers:  # Only add handler if none exists
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            f'%(asctime)s - {rank_suffix}%(levelname)s - %(name)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        handler.setFormatter(formatter)
        root_logger.addHandler(handler)
    
    # Set level for this module's logger and root logger
    logger.setLevel(level)
    root_logger.setLevel(level)
    
    return logger

class CustomPdb(pdb.Pdb):
    """Enhanced PDB with custom prompt."""
    def __init__(self, completekey=None, stdin=None, stdout=None, **kwargs):
        super().__init__(completekey=completekey, stdin=stdin, stdout=stdout, **kwargs)
        self.prompt = '(custom-pdb) '

def setup_socket(path):
    """Create and set up a Unix socket for debugging.
    
    Args:
        path (str): Path where to create the socket
        
    Returns:
        socket.socket: The connected socket
    """
    # Remove existing socket file if present
    if os.path.exists(path):
        os.unlink(path)
        
    # Create server socket
    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server.bind(path)
    server.listen(1)
    
    logger.info(f"Waiting for debugger client to connect on {path}, use 'nc -U {path}' to connect to the debugger.")
    conn, _ = server.accept()
    logger.info("Debugger client connected")
    
    server.close()
    return conn

def get_socket_pdb_params(rank=0):
    """Get parameters for socket-based PDB debugger.
    
    Args:
        rank (int): Process rank for multi-process debugging
        
    Returns:
        dict: Parameters to pass to pdb.Pdb constructor
    """
    # Create socket with rank-specific name
    socket_path = f"{SOCK_PATH}.{rank}"
    conn = setup_socket(socket_path)

    # Wrap socket connection as file objects
    conn_r = conn.makefile('r')
    conn_w = conn.makefile('w')

    # Create parameters dict for pdb
    debugger_params = {
        'stdin': conn_r,
        'stdout': conn_w
    }
    return debugger_params, socket_path


class Debugger:
    """
    Debugger encapsulates debugging behavior for functions.
    """
    base_port = 4444
    # Initialize flag from env var
    debug_flag = os.getenv('IPDB_DEBUG', '').lower() in ('1', 'true', 'yes')
    
    # Debug mode: 'console', 'web', or 'socket'
    debug_mode = os.getenv('IPDB_MODE', '').lower()
    if not debug_mode:
        debug_mode = 'socket'
    elif debug_mode not in ('console', 'web', 'socket'):
        raise ValueError(f"Invalid debug mode: {debug_mode}. Must be one of 'console', 'web', or 'socket'.")
    
    @staticmethod
    def _deep_tb():
        """Return the deepest (innermost) frame and traceback."""
        _, _, tb = sys.exc_info()
        if tb is None:
            return None, None
        while tb.tb_next:
            tb = tb.tb_next
        return tb.tb_frame, tb

    @classmethod
    def web_post_mortem(cls, port=4444, rank=0):
        """Start a web-based post-mortem debugging session, preserving full stack."""
        # Get current exception info
        exc_type, exc_value, exc_tb = sys.exc_info()
        if exc_tb is None:
            logger.error("No traceback to debug")
            return

        # Find the deepest frame in the call stack
        tb_deep = exc_tb
        while tb_deep.tb_next:
            tb_deep = tb_deep.tb_next

        # Get the corresponding frame
        frame_deep = tb_deep.tb_frame

        logger.error(f"Error occurred at [rank:{rank}]: '{cls.exception.__class__.__name__}({cls.exception})'")
        logger.info(f"Starting WebPdb post_mortem server on port {port}...")
        logger.info(f"Open http://0.0.0.0:{port}/ in browser to debug.")

        # Initialize and start post-mortem
        debugger = WebPdb(port=port)
        debugger.reset()
        debugger.interaction(frame_deep, tb_deep)
    
    @classmethod
    def blocking_console_post_mortem(cls, rank=0):
        """
        Start an in-process pdb console and block execution.
        
        Args:
            rank (int): Process rank for distributed debugging
        """
        frame, tb = Debugger._deep_tb()
        if tb is None:
            logger.error("No traceback to debug")
            return
            
        if Debugger.debug_mode == 'socket':
            # Use socket-based debugging
            logger.info(f"Starting socket-based debugging for rank {rank}...")
            
            logger.error(f"Error occurred at [rank:{rank}]: '{cls.exception.__class__.__name__}({cls.exception})'")
            param, socket_path = get_socket_pdb_params(rank=rank)
            p = pdb.Pdb(**param)
            logger.info(f"Connection established on {socket_path}")
            p.prompt = f'(rank-{rank}-pdb) '
        else:
            # Use standard console debugging
            p = pdb.Pdb()
            logger.error(f"Error occurred at [rank:{rank}]: '{cls.exception.__class__.__name__}({cls.exception})'")
            if rank != 0:
                logger.warning(f"Rank {rank} has blocked execution for debugging.")
                while True:
                    pass
            
        p.reset()
        p.interaction(frame, tb)

    @classmethod
    def attach_on_error(cls):
        """
        Decorator that wraps functions with exception debugging capabilities.
        When an exception occurs and debugging is enabled, this decorator will:
        1. Capture the exception context
        2. Start an appropriate debugger based on the environment and configuration
        3. Re-raise the exception after debugging session ends
        
        Returns:
            function: A decorator function that wraps the target function
        """
        def exception_debugging_decorator(target_function):
            """Inner decorator that wraps the target function with exception handling."""
            if not callable(target_function):
                raise TypeError(f"Expected a callable, got {type(target_function).__name__}")
            
            logger.info(f"Registering {target_function.__name__} for debug on error, using `export IPDB_DEBUG=1` to enable debugger auto attach when error occurs.")
            
            def debuggable_function_wrapper(*args, **kwargs):
                """Wrapper that executes the function and handles exceptions with debugging."""
                try:
                    # Execute the original function
                    return target_function(*args, **kwargs)
                except Exception as caught_exception:
                    # Skip debugging if not enabled via flag or environment variable
                    if not cls.debug_flag:
                        raise
                    
                    # Log the exception details
                    logger.error(f"Exception caught in {target_function.__name__}:")
                    traceback.print_exc()
                    cls.exception = caught_exception
                    
                    # Determine process rank for distributed environments
                    process_rank = 0
                    if _dist_available and dist.is_initialized():
                        process_rank = dist.get_rank()
                        logger.debug(f"Detected distributed environment, process rank: {process_rank}")
                    
                    # Start appropriate debugger based on rank and debug mode
                    if process_rank == 0:
                        # Primary process (rank 0) always uses console debugger
                        logger.info("Entering ipdb post_mortem debugger...")
                        ipdb.post_mortem()
                    else:
                        # Non-primary processes use the configured debug mode
                        if cls.debug_mode == 'web':
                            debugger_port = cls.base_port + process_rank
                            logger.info(f"Rank {process_rank} entering web debugger on port {debugger_port}")
                            cls.web_post_mortem(port=debugger_port)
                        elif cls.debug_mode == 'socket':
                            logger.info(f"Rank {process_rank} entering socket-based debugger")
                            cls.blocking_console_post_mortem(rank=process_rank)
                        else:
                            cls.blocking_console_post_mortem(rank=process_rank)
                            # # Default fallback to web post-mortem for non-primary ranks
                            # debugger_port = cls.base_port + process_rank
                            # logger.info(f"Rank {process_rank} entering web debugger (fallback) on port {debugger_port}")
                            # cls.web_post_mortem(port=debugger_port)
                    
                    # Re-raise the exception after debugging session ends
                    raise
                
            return debuggable_function_wrapper
        
        return exception_debugging_decorator


def main():
    parser = argparse.ArgumentParser(description="Debug utils CLI")
    parser.add_argument('--mode', choices=['hello', 'error', 'distributed_error'], default='hello')
    parser.add_argument('--debug', action='store_true', help='Enable debug manually')
    parser.add_argument('--name', type=str, default='World')
    parser.add_argument('--log_level', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                       default='INFO', help='Set logging level')
    args = parser.parse_args()

    # Set up logging
    log_level = getattr(logging, args.log_level)
    rank = None
    if _dist_available and args.mode == 'distributed_error':
        # For distributed mode, try to get rank from env
        if 'RANK' in os.environ:
            rank = int(os.environ['RANK'])
    setup_logging(level=log_level, rank=rank)

    # Manual override for debug flag
    if args.debug:
        Debugger.debug_flag = True
        logger.info("Debug mode enabled via command line flag")
    elif Debugger.debug_flag:
        logger.info("Debug mode enabled via environment variable")
    else:
        logger.info("Debug mode disabled. Use --debug or set IPDB_DEBUG=1 to enable.")

    @Debugger.attach_on_error()
    def hello(name):
        logger.info(f"Hello, {name}!")

    @Debugger.attach_on_error()
    def error():
        logger.warning("About to generate a test error")
        raise RuntimeError("Test error for Debugger")

    @Debugger.attach_on_error()
    def distributed_error():
        if _dist_available and dist.is_initialized():
            dist.barrier()
            rank = dist.get_rank()
            logger.info(f"Running on rank {rank}")
        else:
            rank = 0
        logger.warning(f"About to generate an error on rank {rank}")
        raise RuntimeError(f"Error on rank {rank}")

    # Distributed init for torchrun
    if args.mode == 'distributed_error' and _dist_available:
        backend = 'nccl' if torch.cuda.is_available() else 'gloo'
        try:
            logger.debug(f"Initializing process group with backend: {backend}")
            dist.init_process_group(backend=backend)
            # Update rank now that we're initialized
            rank = dist.get_rank()
            setup_logging(level=log_level, rank=rank)
            logger.info(f"Process group initialized, rank: {rank}, world size: {dist.get_world_size()}")
        except ValueError as e:
            logger.error(f"Failed to init process group: {e}")
            logger.error("Set MASTER_ADDR, MASTER_PORT, WORLD_SIZE, RANK env vars or specify init_method.")
            return

    # Dispatch
    if args.mode == 'hello':
        hello(args.name)
    elif args.mode == 'error':
        try:
            error()
        except Exception as e:
            logger.error(f"Error caught in main: {e}")
    else:
        try:
            distributed_error()
        except Exception as e:
            logger.error(f"Error caught in main: {e}")

    # Cleanup
    if args.mode == 'distributed_error' and _dist_available and dist.is_initialized():
        logger.debug("Cleaning up process group")
        dist.destroy_process_group()
        logger.info("Process group destroyed")

if __name__ == '__main__':
    main()
