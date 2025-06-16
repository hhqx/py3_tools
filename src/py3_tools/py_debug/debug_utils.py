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
    
    print(f"[main] Waiting for debugger client to connect on {path}")
    conn, _ = server.accept()
    print("[main] Debugger client connected")
    
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
    conn = setup_socket(f"{SOCK_PATH}.{rank}")
    print(f"[get_param] Connection established on {SOCK_PATH}.{rank}")

    # Wrap socket connection as file objects
    conn_r = conn.makefile('r')
    conn_w = conn.makefile('w')

    # Create parameters dict for pdb
    debugger_params = {
        'stdin': conn_r,
        'stdout': conn_w
    }
    return debugger_params


class Debugger:
    """
    Debugger encapsulates debugging behavior for functions.
    """
    base_port = 4444
    # Initialize flag from env var
    debug_flag = os.getenv('IPDB_DEBUG', '').lower() in ('1', 'true', 'yes')
    
    # Debug mode: 'console', 'web', or 'socket'
    debug_mode = os.getenv('IPDB_MODE', 'console')
    
    @staticmethod
    def _deep_tb():
        """Return the deepest (innermost) frame and traceback."""
        _, _, tb = sys.exc_info()
        if tb is None:
            return None, None
        while tb.tb_next:
            tb = tb.tb_next
        return tb.tb_frame, tb

    @staticmethod
    def web_post_mortem(port=4444):
        """Start a web-based post-mortem debugging session, preserving full stack."""
        # 获取当前异常信息
        exc_type, exc_value, exc_tb = sys.exc_info()
        if exc_tb is None:
            print("No traceback to debug")
            return

        # 找到调用链最深（出错点）的 tb
        tb_deep = exc_tb
        while tb_deep.tb_next:
            tb_deep = tb_deep.tb_next

        # 取出对应的 frame
        frame_deep = tb_deep.tb_frame

        print(f"Starting WebPdb post_mortem server on port {port}...")
        print(f"Open http://0.0.0.0:{port}/ in browser to debug.")

        # 初始化并启动 post-mortem
        debugger = WebPdb(port=port)
        debugger.reset()                       # 重置老状态
        debugger.interaction(frame_deep, tb_deep)  # 从最深帧开始调试
    
    @staticmethod
    def blocking_console_post_mortem(rank=0):
        """
        Start an in-process pdb console and block execution.
        
        Args:
            rank (int): Process rank for distributed debugging
        """
        frame, tb = Debugger._deep_tb()
        if tb is None:
            print("No traceback to debug")
            return
            
        if Debugger.debug_mode == 'socket':
            # Use socket-based debugging
            print(f"Starting socket-based debugging for rank {rank}...")
            
            logger.error(f"Error occured at [rank:{rank}], using 'nc -U {SOCK_PATH}.{rank}' to connect to the debugger.")
            p = pdb.Pdb(**get_socket_pdb_params(rank=rank))
            p.prompt = f'(rank-{rank}-pdb) '
        else:
            # Use standard console debugging
            p = pdb.Pdb()
            if rank != 0:
                while True:
                    # logger.info(f"Rank {rank} has blocked execution for debugging. ")
                    pass
            
        p.reset()
        p.interaction(frame, tb)

    @classmethod
    def on_error(cls):
        """Decorator to wrap functions for debug on exception."""
        def decorator(func):
            def wrapper(*args, **kwargs):
                try:
                    return func(*args, **kwargs)
                except Exception:
                    # Trigger if env var or manual flag
                    if not cls.debug_flag:
                        raise
                    traceback.print_exc()
                    rank = 0
                    if _dist_available and dist.is_initialized():
                        rank = dist.get_rank()
                        
                    # Handle different debug modes
                    if rank == 0:
                        ipdb.post_mortem()
                    else:
                        if cls.debug_mode == 'web':
                            port = cls.base_port + rank
                            cls.web_post_mortem(port=port)
                        elif cls.debug_mode == 'socket':
                            # Use socket-based or console-based debugging
                            cls.blocking_console_post_mortem(rank=rank)
                        else:
                            # Default fallback to web post-mortem
                            port = cls.base_port + rank
                            cls.web_post_mortem(port=port)
                    raise
            return wrapper
        return decorator


def main():
    parser = argparse.ArgumentParser(description="Debug utils CLI")
    parser.add_argument('--mode', choices=['hello', 'error', 'distributed_error'], default='hello')
    parser.add_argument('--debug', action='store_true', help='Enable debug manually')
    parser.add_argument('--name', type=str, default='World')
    args = parser.parse_args()

    # Manual override
    if args.debug:
        Debugger.debug_flag = True

    @Debugger.on_error()
    def hello(name):
        print(f"Hello, {name}!")

    @Debugger.on_error()
    def error():
        raise RuntimeError("Test error for Debugger")

    @Debugger.on_error()
    def distributed_error():
        if _dist_available and dist.is_initialized():
            dist.barrier()
            rank = dist.get_rank()
        else:
            rank = 0
        raise RuntimeError(f"Error on rank {rank}")

    # Distributed init for torchrun
    if args.mode == 'distributed_error' and _dist_available:
        backend = 'nccl' if torch.cuda.is_available() else 'gloo'
        try:
            dist.init_process_group(backend=backend)
        except ValueError as e:
            print(f"Failed to init process group: {e}\n"
                  "Set MASTER_ADDR, MASTER_PORT, WORLD_SIZE, RANK env vars or specify init_method.")
            return

    # Dispatch
    if args.mode == 'hello':
        hello(args.name)
    elif args.mode == 'error':
        error()
    else:
        distributed_error()

    # Cleanup
    if args.mode == 'distributed_error' and _dist_available and dist.is_initialized():
        dist.destroy_process_group()

if __name__ == '__main__':
    main()
