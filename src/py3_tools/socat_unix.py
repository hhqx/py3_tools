# 用 click 实现命令行工具
import click
import os
import socket
import sys
import termios
import tty
import select
import fcntl
import errno
import signal
import time
from contextlib import contextmanager


@contextmanager
def raw_terminal():
    """Set the terminal to raw mode and restore it after use."""
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        # Set terminal to raw mode
        tty.setraw(fd)
        yield
    finally:
        # Restore terminal settings
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        # Print newline to ensure prompt appears correctly after exit
        sys.stdout.write('\r\n')
        sys.stdout.flush()


def make_non_blocking(fd):
    """Set file descriptor to non-blocking mode."""
    flags = fcntl.fcntl(fd, fcntl.F_GETFL)
    fcntl.fcntl(fd, fcntl.F_SETFL, flags | os.O_NONBLOCK)


class SocketConnector:
    """Handle connection to UNIX socket and data transfer."""
    
    def __init__(self, socket_path):
        self.socket_path = socket_path
        self.sock = None
        self.running = False
        self.input_buffer = bytearray()
        self.exit_command = b'exit\r'
        
    def connect(self):
        """Connect to the UNIX socket."""
        self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            self.sock.connect(self.socket_path)
            return True
        except socket.error as e:
            click.echo(f"Failed to connect to socket {self.socket_path}: {e}")
            return False
            
    def handle_io(self):
        """Handle bidirectional communication between terminal and socket."""
        stdin_fd = sys.stdin.fileno()
        stdout_fd = sys.stdout.fileno()
        
        # Make socket non-blocking
        make_non_blocking(self.sock.fileno())
        
        self.running = True
        
        # Setup signal handler for clean exit
        def signal_handler(sig, frame):
            self.running = False
        
        signal.signal(signal.SIGINT, signal_handler)
        
        # Main I/O loop
        while self.running:
            # Use select to wait for available data
            readable, _, exceptional = select.select(
                [stdin_fd, self.sock], [], [stdin_fd, self.sock], 0.1)
            
            # Handle keyboard input
            if stdin_fd in readable:
                try:
                    data = os.read(stdin_fd, 1024)
                    if not data:  # EOF
                        self.running = False
                        break
                    
                    # Add data to the input buffer and check for exit command
                    self.input_buffer.extend(data)
                    
                    # Check if buffer ends with the exit command
                    if len(self.input_buffer) >= len(self.exit_command) and \
                       self.input_buffer[-len(self.exit_command):] == self.exit_command:
                        # Display a graceful exit message
                        exit_msg = b'\r\nExiting session...\r\n'
                        os.write(stdout_fd, exit_msg)
                        self.running = False
                        break
                    
                    # Keep buffer from growing too large
                    if len(self.input_buffer) > 256:
                        self.input_buffer = self.input_buffer[-256:]
                    
                    # Send data to socket
                    self.sock.sendall(data)
                except (OSError, socket.error) as e:
                    if e.errno != errno.EAGAIN and e.errno != errno.EWOULDBLOCK:
                        click.echo(f"\r\nError reading from stdin: {e}")
                        self.running = False
                        break
            
            # Handle socket data
            if self.sock in readable:
                try:
                    data = self.sock.recv(1024)
                    if not data:  # Socket closed
                        click.echo("\r\nConnection closed by peer")
                        self.running = False
                        break
                    os.write(stdout_fd, data)
                except socket.error as e:
                    if e.errno != errno.EAGAIN and e.errno != errno.EWOULDBLOCK:
                        click.echo(f"\r\nError reading from socket: {e}")
                        self.running = False
                        break
            
            # Handle exceptions
            if exceptional:
                click.echo("\r\nException occurred")
                self.running = False
                break
    
    def close(self):
        """Close the socket connection."""
        if self.sock:
            self.sock.close()


@click.command()
@click.option('--socket_path', '-s', required=True, help='Path to the UNIX socket')
def socat_unix(socket_path):
    """Connect to a UNIX socket and provide interactive terminal access.
    
    This is a Python replacement for: socat $(tty),raw,echo=0 UNIX-CONNECT:{socket_path}
    
    Type 'exit' to quit the session.
    """
    # Check if socket path exists
    if not os.path.exists(socket_path):
        click.echo(f"Socket path does not exist: {socket_path}")
        return
    
    connector = SocketConnector(socket_path)
    
    # Try to connect
    if not connector.connect():
        return
    
    click.echo(f"Connected to {socket_path}")
    click.echo("Press Ctrl+C or type 'exit' to quit")
    
    try:
        with raw_terminal():
            connector.handle_io()
    except Exception as e:
        click.echo(f"\r\nError: {e}")
    finally:
        connector.close()


if __name__ == "__main__":
    socat_unix()  # Using command name that matches module name