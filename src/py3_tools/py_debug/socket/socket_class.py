import socket
import sys
import os
import pty
import termios
import select
import threading
import time
import io
import enum
import logging
import tempfile
import ipdb
import atexit
from typing import Optional, Union, Tuple, TextIO, BinaryIO, Dict, Any, Callable
import uuid

# Configure logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

__ALL__ = [
    'ConnectionType', 'CustomSocketPdb', 
    'socket_set_trace', 'socket_post_mortem', 
    'remote_set_trace', 'remote_set_trace_unix',
    'DEFAULT_SOCK_PATH'
]

# Define default socket path that can be imported by other modules
DEFAULT_SOCK_PATH = os.path.join(tempfile.gettempdir(), f'pdb.sock.{os.getpid()}-{str(uuid.uuid4())[-8:]}')

class ConnectionType(enum.Enum):
    """Enumeration of supported connection types for socket debugging."""
    TCP = "tcp"
    UNIX = "unix"

def get_rank() -> int:
    # Get rank from environment variable, default to -1 if not found or invalid
    try:
        rank = int(os.environ.get('RANK', '-1'))
    except ValueError:
        logger.warning("Invalid RANK environment variable, defaulting to -1")
        rank = -1
    return rank

# Global variable to store active debugger instances
_active_debuggers = {}

class CustomSocketPdb:
    """
    Custom class for remote debugging through sockets.
    
    Supports both TCP and Unix domain sockets for remote connections,
    allowing flexible debugging options depending on the deployment scenario.
    
    Attributes:
        connection_type: Type of connection (TCP or UNIX socket)
        host: Host to listen on for TCP connections
        port: Port to listen on for TCP connections
        unix_socket_path: Path for Unix domain socket
        server: Server socket instance
        client_socket: Client socket connection
        master_fd: Master file descriptor for pseudo-terminal
        slave_fd: Slave file descriptor for pseudo-terminal
        socket_thread: Thread for relaying data from socket to pty
        pty_thread: Thread for relaying data from pty to socket
    """
    
    def __init__(self, connection_type: ConnectionType = ConnectionType.TCP, 
                 host: str = 'localhost', port: int = 5678, 
                 unix_socket_path: Optional[str] = None,
                 **kwargs):
        """
        Initialize a CustomSocketPdb instance.
        
        Args:
            connection_type: Type of connection to use (TCP or UNIX)
            host: Host to listen on for TCP connections (default: 'localhost')
            port: Port to listen on for TCP connections (default: 5678)
            unix_socket_path: Path for Unix domain socket (default: auto-generated in /tmp)
            **kwargs: Additional keyword arguments
        """
        self.connection_type = connection_type
        self.host = host
        self.port = port
        
        # Auto-generate unix socket path if not provided
        if unix_socket_path is None:
            self.unix_socket_path = os.path.join(tempfile.gettempdir(), f"debugger_{os.getpid()}.sock")
        else:
            self.unix_socket_path = unix_socket_path
            
        # Initialize instance variables
        self.server = None
        self.client_socket = None
        self.master_fd = None
        self.slave_fd = None
        self.slave_file = None
        self.socket_thread = None
        self.pty_thread = None
        self.is_active = False
        
        # Store original streams
        self.orig_stdin = sys.stdin
        self.orig_stdout = sys.stdout
        self.orig_stderr = sys.stderr
        
        self.connect_command = None

    def start(self) -> None:
        """Start the socket server and set up the debugging environment."""
        # If already active, just return
        if self.is_active:
            logger.debug("Debugger already active, reusing existing connection")
            return

        try:
            # Set up the server based on connection type
            self._setup_server()
            
            # Accept client connection
            self._accept_client()
            
            # Create pseudo-terminal
            self._setup_pty()
            
            # Redirect stdin/stdout/stderr to the pty
            sys.stdin = self.slave_file
            sys.stdout = self.slave_file
            sys.stderr = self.slave_file
            
            # Start relay threads
            self._start_relay_threads()
            
            # Configure terminal settings
            self._setup_terminal()
            
            # Give a moment for connections to stabilize
            time.sleep(0.1)
            
            # Mark as active
            self.is_active = True
            
            # Register cleanup on exit
            atexit.register(self.cleanup)
            
        except Exception as e:
            self.cleanup()
            raise RuntimeError(f"Failed to set up debugger connection: {e}") from e

    def _setup_server(self) -> None:
        """Set up the server socket based on the connection type."""
        if self.connection_type == ConnectionType.TCP:
            self._setup_tcp_server()
        else:  # UNIX
            self._setup_unix_server()
    
    def _setup_tcp_server(self) -> None:
        """Set up a TCP server socket."""
        self.server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        
        try:
            self.server.bind((self.host, self.port))
            self.server.listen(1)
            
            logger.info(f"[*] Remote debugger waiting for TCP connection at {self.host}:{self.port}")
            
            rank = get_rank()
                
            # Create a visually appealing box with connection instructions
            connect_cmd = f"socat $(tty),raw,echo=0 TCP:{self.host}:{self.port}"
            title = "DEBUGGER READY - CONNECTION INSTRUCTIONS"
            if rank >=0:
                title = f"RANK {rank}: {title}"
            logger.info(f"[rank={rank}] Connect Debugger with: '{connect_cmd}'")
            
            # Calculate box width based on the longest content (without ANSI codes)
            box_width = max(len(connect_cmd), len(title)) + 4  # +4 for padding
            
            # Create highlighted command with ANSI codes
            highlighted_cmd = f"\033[1m\033[32m{connect_cmd}\033[0m"
            highlighted_title = f"\033[1m\033[33m{title}\033[0m"
            
            # Create the colorful box with connection instructions
            box = [
                f"\033[1m\033[34m{'╔' + '═' * (box_width-2) + '╗'}\033[0m",
                f"\033[1m\033[34m║\033[0m{highlighted_title.center(box_width-2+len(highlighted_title)-len(title))}\033[1m\033[34m║\033[0m",
                f"\033[1m\033[34m╟{'─' * (box_width-2)}╢\033[0m",
                f"\033[1m\033[34m║\033[0m{highlighted_cmd.center(box_width-2+len(highlighted_cmd)-len(connect_cmd))}\033[1m\033[34m║\033[0m",
                f"\033[1m\033[34m{'╚' + '═' * (box_width-2) + '╝'}\033[0m"
            ]
            
            # Print the box
            for line in box:
                # logger.info(line)
                print(line)  # Print to stdout as well
            
            # Store the connection command for reference
            self.connect_command = connect_cmd
        except OSError as e:
            raise OSError(f"Failed to bind to {self.host}:{self.port}: {e}")
    
    def _setup_unix_server(self) -> None:
        """Set up a Unix domain socket server."""
        # Clean up existing socket file if present
        if os.path.exists(self.unix_socket_path):
            try:
                os.unlink(self.unix_socket_path)
            except OSError as e:
                raise OSError(f"Failed to remove existing socket file {self.unix_socket_path}: {e}")
        
        self.server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        
        try:
            self.server.bind(self.unix_socket_path)
            self.server.listen(1)
            
            # Make socket file accessible
            try:
                os.chmod(self.unix_socket_path, 0o750)
            except OSError:
                logger.warning(f"Could not change permissions of {self.unix_socket_path}")
            
            logger.info(f"[*] Remote debugger waiting for UNIX socket connection at {self.unix_socket_path}")
            
            rank = get_rank()
                
            # Create a visually appealing box with connection instructions
            connect_cmd = f"socat $(tty),raw,echo=0 UNIX-CONNECT:{self.unix_socket_path}"
            title = "DEBUGGER READY - CONNECTION INSTRUCTIONS"
            if rank >= 0:
                title = f"RANK {rank}: {title}"
            
            logger.info(f"[rank={rank}] Connect Debugger with: '{connect_cmd}'")
            
            # Calculate box width based on the longest content (without ANSI codes)
            box_width = max(len(connect_cmd), len(title)) + 4  # +4 for padding
            
            # Create highlighted command with ANSI codes
            highlighted_cmd = f"\033[1m\033[32m{connect_cmd}\033[0m"
            highlighted_title = f"\033[1m\033[33m{title}\033[0m"
            
            # Create the colorful box with connection instructions
            box = [
                f"\033[1m\033[34m{'╔' + '═' * (box_width-2) + '╗'}\033[0m",
                f"\033[1m\033[34m║\033[0m{highlighted_title.center(box_width-2+len(highlighted_title)-len(title))}\033[1m\033[34m║\033[0m",
                f"\033[1m\033[34m╟{'─' * (box_width-2)}╢\033[0m",
                f"\033[1m\033[34m║\033[0m{highlighted_cmd.center(box_width-2+len(highlighted_cmd)-len(connect_cmd))}\033[1m\033[34m║\033[0m",
                f"\033[1m\033[34m{'╚' + '═' * (box_width-2) + '╝'}\033[0m"
            ]
            
            # Print the box
            for line in box:
                # logger.info(line)
                print(line)
            
            self.connect_command = connect_cmd
        except OSError as e:
            raise OSError(f"Failed to bind to {self.unix_socket_path}: {e}")

    def _accept_client(self) -> None:
        """Accept a client connection to the server socket."""
        try:
            self.client_socket, addr = self.server.accept()
            if self.connection_type == ConnectionType.TCP:
                logger.info(f"[*] Connection from {addr[0]}:{addr[1]}")
            else:
                logger.info("[*] Client connected to UNIX socket")
        except OSError as e:
            raise OSError(f"Error accepting connection: {e}")

    def _setup_pty(self) -> None:
        """Create a pseudo-terminal and set up file wrappers."""
        self.master_fd, self.slave_fd = pty.openpty()
        
        # Create a custom TextIOWrapper with encoding
        class EncodedIO(io.TextIOWrapper):
            def __init__(self, fd, **kwargs):
                # Open the file descriptor in binary mode
                binary_io = os.fdopen(fd, 'rb+', buffering=0)
                # Initialize TextIOWrapper with encoding
                super().__init__(binary_io, encoding='utf-8', errors='replace', write_through=True, **kwargs)
        
        # Open slave fd with encoding
        self.slave_file = EncodedIO(self.slave_fd)
        
    def _socket_to_pty(self) -> None:
        """Relay data from socket to pseudo-terminal."""
        while True:
            try:
                data = self.client_socket.recv(1024)
                if not data:
                    break
                os.write(self.master_fd, data)
            except (OSError, BrokenPipeError) as e:
                logger.debug(f"Socket to PTY error: {e}")
                break
            except Exception as e:
                logger.error(f"Socket to PTY error: {e}")
                break
    
    def _pty_to_socket(self) -> None:
        """Relay data from pseudo-terminal to socket."""
        while True:
            try:
                r, _, _ = select.select([self.master_fd], [], [], 0.1)
                if self.master_fd in r:
                    data = os.read(self.master_fd, 1024)
                    if not data:
                        break
                    self.client_socket.sendall(data)
            except (OSError, BrokenPipeError) as e:
                logger.debug(f"PTY to socket error: {e}")
                break
            except Exception as e:
                logger.error(f"PTY to socket error: {e}")
                break

    def _start_relay_threads(self) -> None:
        """Start threads to relay data between socket and pseudo-terminal."""
        self.socket_thread = threading.Thread(target=self._socket_to_pty, daemon=True)
        self.pty_thread = threading.Thread(target=self._pty_to_socket, daemon=True)
        
        self.socket_thread.start()
        self.pty_thread.start()

    def _setup_terminal(self) -> None:
        """Configure terminal settings for the pseudo-terminal."""
        try:
            # Initialize terminal settings
            term_settings = termios.tcgetattr(self.slave_fd)
            term_settings[3] = term_settings[3] & ~termios.ECHO  # Disable echo
            termios.tcsetattr(self.slave_fd, termios.TCSANOW, term_settings)
        except (termios.error, OSError) as e:
            logger.warning(f"Failed to configure terminal settings: {e}")

    def cleanup(self) -> None:
        """Clean up resources (pipes, sockets, file descriptors)."""
        # Skip if not active
        if not self.is_active:
            return
            
        logger.debug("Cleaning up debugger resources")
        
        # Restore original streams
        if hasattr(self, 'orig_stdin') and self.orig_stdin:
            sys.stdin = self.orig_stdin
        if hasattr(self, 'orig_stdout') and self.orig_stdout:
            sys.stdout = self.orig_stdout
        if hasattr(self, 'orig_stderr') and self.orig_stderr:
            sys.stderr = self.orig_stderr
        
        # Close slave file if open
        if hasattr(self, 'slave_file') and self.slave_file:
            try:
                self.slave_file.close()
            except Exception:
                pass
            self.slave_file = None
        
        # Close master fd if open
        if hasattr(self, 'master_fd') and self.master_fd is not None:
            try:
                os.close(self.master_fd)
            except Exception:
                pass
            self.master_fd = None
        
        # Close client socket if open
        if hasattr(self, 'client_socket') and self.client_socket:
            try:
                self.client_socket.close()
            except Exception:
                pass
            self.client_socket = None
        
        # Close server socket if open
        if hasattr(self, 'server') and self.server:
            try:
                self.server.close()
            except Exception:
                pass
            self.server = None
            
        # Remove Unix socket file if it exists
        if hasattr(self, 'connection_type') and self.connection_type == ConnectionType.UNIX and \
           hasattr(self, 'unix_socket_path') and os.path.exists(self.unix_socket_path):
            try:
                os.unlink(self.unix_socket_path)
            except Exception:
                pass
                
        # Remove from active debuggers
        key = self._get_instance_key()
        if key in _active_debuggers:
            del _active_debuggers[key]
            
        # Mark as inactive
        self.is_active = False
        
        # Unregister from atexit
        try:
            atexit.unregister(self.cleanup)
        except Exception:
            pass

    def _get_instance_key(self) -> str:
        """Get unique key for this debugger instance based on connection details."""
        if self.connection_type == ConnectionType.TCP:
            return f"tcp:{self.host}:{self.port}"
        else:
            return f"unix:{self.unix_socket_path}"

    def __enter__(self):
        """Support context manager protocol."""
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Clean up resources when exiting context manager."""
        self.cleanup()


def get_or_create_debugger(connection_type=ConnectionType.TCP, host='localhost', 
                           port=5678, unix_socket_path=None) -> CustomSocketPdb:
    """
    Get an existing debugger instance or create a new one if none exists.
    
    This ensures we don't create multiple debuggers for the same connection.
    
    Args:
        connection_type: Type of connection (TCP or UNIX)
        host: Host for TCP connections
        port: Port for TCP connections
        unix_socket_path: Path for Unix socket
        
    Returns:
        CustomSocketPdb: An existing or new debugger instance
    """
    # Generate key for lookup
    if connection_type == ConnectionType.TCP:
        key = f"tcp:{host}:{port}"
    else:
        if unix_socket_path is None:
            unix_socket_path = os.path.join(tempfile.gettempdir(), f"debugger_{os.getpid()}.sock")
        key = f"unix:{unix_socket_path}"
    
    # Check if we have an active debugger for this connection
    if key in _active_debuggers and _active_debuggers[key].is_active:
        logger.debug(f"Reusing existing debugger for {key}")
        return _active_debuggers[key]
    
    # Create new debugger
    debugger = CustomSocketPdb(
        connection_type=connection_type,
        host=host,
        port=port,
        unix_socket_path=unix_socket_path
    )
    
    # Store it for future reuse
    _active_debuggers[key] = debugger
    return debugger


def socket_set_trace(frame=None, connection_type=ConnectionType.TCP, host='localhost', 
                   port=5678, unix_socket_path=None):
    """
    Start a debugging session through a socket with ipdb.
    
    Args:
        frame: The frame to debug (default: caller's frame)
        connection_type: Type of connection (TCP or UNIX)
        host: Host to listen on for TCP connections (default: 'localhost')
        port: Port to listen on for TCP connections (default: 5678)
        unix_socket_path: Path for Unix domain socket (default: auto-generated in /tmp)
    
    Usage:
        1. Add this function to your code
        2. Call socket_set_trace() at any point where you want to debug
        3. When code reaches this point, connect from another terminal:
           - For TCP: socat $(tty),raw,echo=0 TCP:localhost:5678
           - For Unix socket: socat $(tty),raw,echo=0 UNIX-CONNECT:<socket_path>
    """
    if frame is None:
        frame = sys._getframe().f_back
    
    # Get or create debugger instance
    debugger = get_or_create_debugger(
        connection_type=connection_type,
        host=host,
        port=port,
        unix_socket_path=unix_socket_path
    )
    
    try:
        # Start the debugger if not already started
        debugger.start()
    except Exception as e:
        logger.error(f"Failed to start debugger: {e}")
        debugger.cleanup()
        return
    
    # Use ipdb with the redirected streams
    ipdb.set_trace(frame=frame)


def socket_post_mortem(tb=None, connection_type=ConnectionType.TCP, host='localhost',
                     port=5678, unix_socket_path=None):
    """
    Start a post-mortem debugging session through a socket.
    
    Args:
        tb: Traceback object to debug (default: current exception)
        connection_type: Type of connection (TCP or UNIX)
        host: Host to listen on for TCP connections (default: 'localhost')
        port: Port to listen on for TCP connections (default: 5678)
        unix_socket_path: Path for Unix domain socket (default: auto-generated in /tmp)
        
    Usage:
        try:
            # Some code that might raise an exception
            result = 1/0
        except Exception:
            socket_post_mortem()
    """
    if tb is None:
        # Get current exception
        _, _, tb = sys.exc_info()
    
    if tb is None:
        raise ValueError("No traceback provided and no exception is being handled")
    
    # Get or create debugger instance
    debugger = get_or_create_debugger(
        connection_type=connection_type,
        host=host,
        port=port,
        unix_socket_path=unix_socket_path
    )
    
    try:
        # Start the debugger if not already started
        debugger.start()
    except Exception as e:
        logger.error(f"Failed to start debugger: {e}")
        debugger.cleanup()
        return
    
    # Use ipdb post_mortem with the redirected streams
    ipdb.post_mortem(tb)

# Ensure all resources are cleaned up at process exit
def cleanup_all_debuggers():
    """Clean up all active debuggers when the process exits."""
    for key, debugger in list(_active_debuggers.items()):
        try:
            debugger.cleanup()
        except Exception as e:
            logger.debug(f"Error cleaning up debugger {key}: {e}")

# Register the cleanup function
atexit.register(cleanup_all_debuggers)

def remote_set_trace(host='localhost', port=5678, connection_type=ConnectionType.TCP, 
                unix_socket_path=None):
    """
    Set up a remote debugging session with proper terminal support.
    
    This function maintains backward compatibility with the original implementation.
    
    When execution reaches this point, it will wait for a remote connection
    before starting the debugger.
    
    Args:
        host (str): Host to listen on for TCP connections (default: 'localhost')
        port (int): Port to listen on for TCP connections (default: 5678)
        connection_type (ConnectionType): Type of connection (TCP or UNIX)
        unix_socket_path (str, optional): Path for Unix domain socket
    
    Usage:
        1. Add this function to your code
        2. Insert `remote_trace()` at any point where you want to debug
        3. When code reaches this point, connect from another terminal:
           - For TCP: socat -,raw,echo=0 TCP:localhost:5678
           - For Unix socket: socat -,raw,echo=0 UNIX-CONNECT:/tmp/debugger.sock
    """
    # Get the calling frame
    frame = sys._getframe().f_back
    
    # Start debugging
    socket_set_trace(
        frame=frame,
        connection_type=connection_type,
        host=host,
        port=port,
        unix_socket_path=unix_socket_path
    )


def remote_set_trace_unix(socket_path=None):
    """
    Set up a remote debugging session using Unix domain sockets.
    
    This is a convenience wrapper around remote_trace() that defaults
    to using Unix domain sockets.
    
    Args:
        socket_path (str, optional): Path for Unix domain socket. If None,
                                    a default path will be generated.
    
    Usage:
        1. Add this function to your code
        2. Insert `remote_trace_unix()` at any point where you want to debug
        3. When code reaches this point, connect from another terminal:
           socat -,raw,echo=0 UNIX-CONNECT:<socket_path>
    """
    return remote_set_trace(
        connection_type=ConnectionType.UNIX,
        unix_socket_path=socket_path
    )


def example_function():
    """Example function demonstrating remote debugging."""
    a = 1
    b = 2
    remote_set_trace()  # Execution will pause here waiting for connection
    c = a + b
    return c


def example_function_unix():
    """Example function demonstrating Unix socket-based remote debugging."""
    a = 1
    b = 2
    remote_set_trace_unix()  # Execution will pause here waiting for connection
    c = a + b
    return c


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Remote debugging example')
    parser.add_argument('--mode', choices=['tcp', 'unix'], default='tcp',
                        help='Connection type (tcp or unix)')
    parser.add_argument('--host', default='localhost',
                        help='Host for TCP connection')
    parser.add_argument('--port', type=int, default=5678,
                        help='Port for TCP connection')
    parser.add_argument('--socket', help='Unix socket path')
    
    args = parser.parse_args()
    
    if args.mode == 'tcp':
        print(f"Running TCP example on {args.host}:{args.port}")
        result = example_function()
    else:  # unix
        print(f"Running Unix socket example on {args.socket or '<auto>'}")
        result = example_function_unix()
        
    print(f"Result: {result}")

'''
# Connection instructions:

## TCP Socket:
socat -,raw,echo=0 TCP:localhost:5678
socat $(tty),raw,echo=0 TCP:localhost:5678

## Unix Socket:
socat -,raw,echo=0 UNIX-CONNECT:/tmp/debugger_[PID].sock
socat $(tty),raw,echo=0 UNIX-CONNECT:/tmp/debugger_[PID].sock
'''