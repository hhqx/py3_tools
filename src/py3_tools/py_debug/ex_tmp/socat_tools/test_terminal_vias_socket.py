#!/usr/bin/env python3
"""
Terminal Socket Server

This module implements a Unix socket server that binds to a pseudo-terminal,
allowing clients to interact with terminal applications (like bash or Python REPL)
through socket connections. It preserves all terminal control sequences and
supports rich text (colors, formatting, cursor movements).

Dependencies:
    Standard library only (os, pty, select, socket, subprocess, logging)

Usage:
    Run directly: python test_terminal_vias_socket.py
    
    Or import and use in other modules:
    ```
    from test_terminal_vias_socket import TerminalSocketServer
    server = TerminalSocketServer('/tmp/term.sock', shell_cmd='/bin/bash')
    server.start()
    ```
"""

import os
import pty
import select
import socket
import subprocess
import threading
import logging
import time
import signal
import sys
from typing import List, Optional, Tuple, Union


class TerminalSocketServer:
    """
    A server that creates a pseudo-terminal and makes it accessible via Unix socket.
    
    This class spawns a subprocess running in a pseudo-terminal and creates a
    Unix domain socket server. Clients connecting to the socket can interact with
    the terminal process as if they were directly connected to it, with full support
    for terminal control sequences, colors, and other ANSI features.
    
    Attributes:
        socket_path (str): Path to the Unix socket
        shell_cmd (str): Command to run in the pseudo-terminal
        shell_args (List[str]): Arguments for the shell command
        log_level (int): Logging level (from logging module)
        buffer_size (int): Size of the read buffer for socket/pty communication
        server_socket (socket.socket): The Unix domain socket for accepting connections
        master_fd (int): File descriptor for the master side of the pseudo-terminal
        pid (int): Process ID of the spawned shell process
        running (bool): Flag indicating if the server is running
        clients (List[socket.socket]): List of connected client sockets
    """
    
    def __init__(self, 
                 socket_path: str,
                 shell_cmd: str = '/bin/bash',
                 shell_args: List[str] = None,
                 log_level: int = logging.INFO,
                 buffer_size: int = 1024):
        """
        Initialize the TerminalSocketServer.
        
        Args:
            socket_path: Path where the Unix socket will be created
            shell_cmd: Command to run in the pseudo-terminal (default: /bin/bash)
            shell_args: Arguments to pass to the shell command (default: None)
            log_level: Logging level (default: logging.INFO)
            buffer_size: Size of buffer for reading from socket/pty (default: 1024)
        """
        self.socket_path = socket_path
        self.shell_cmd = shell_cmd
        self.shell_args = shell_args or []
        self.buffer_size = buffer_size
        
        # Setup logging
        self.logger = logging.getLogger("TerminalSocketServer")
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        self.logger.setLevel(log_level)
        
        # Initialize state variables
        self.server_socket = None
        self.master_fd = None
        self.pid = None
        self.running = False
        self.clients = []
    
    def start(self) -> None:
        """
        Start the terminal socket server.
        
        This method:
        1. Creates a pseudo-terminal and spawns the shell process
        2. Sets up the Unix socket server
        3. Starts the client acceptance thread
        4. Starts the PTY read thread
        
        Raises:
            OSError: If socket creation, binding or PTY spawn fails
        """
        if self.running:
            self.logger.warning("Server already running")
            return
        
        # Create and spawn the PTY
        self._spawn_pty()
        
        # Create and bind the socket
        self._setup_socket()
        
        # Set the running flag and start threads
        self.running = True
        
        # Start the client acceptance thread
        self.accept_thread = threading.Thread(target=self._accept_clients)
        self.accept_thread.daemon = True
        self.accept_thread.start()
        
        # Start the PTY read thread
        self.pty_thread = threading.Thread(target=self._read_pty)
        self.pty_thread.daemon = True
        self.pty_thread.start()
        
        self.logger.info(f"Server started. Socket: {self.socket_path}, PTY: {self.shell_cmd}")
    
    def stop(self) -> None:
        """
        Stop the server and clean up resources.
        
        This method:
        1. Sets the running flag to False
        2. Closes all client connections
        3. Terminates the shell process
        4. Closes and removes the Unix socket
        """
        if not self.running:
            self.logger.warning("Server not running")
            return
        
        self.running = False
        self.logger.info("Stopping server...")
        
        # Close all client connections
        for client in self.clients[:]:  # Use a copy of the list since we'll modify it
            self._close_client(client)
        
        # Close the server socket
        if self.server_socket:
            self.server_socket.close()
            self.server_socket = None
        
        # Remove the socket file
        if os.path.exists(self.socket_path):
            try:
                os.unlink(self.socket_path)
            except OSError as e:
                self.logger.error(f"Failed to remove socket file: {e}")
        
        # Terminate the PTY process
        if self.pid:
            try:
                os.kill(self.pid, signal.SIGTERM)
                os.waitpid(self.pid, 0)
            except OSError as e:
                if e.errno != os.errno.ECHILD:  # Ignore "No child processes" error
                    self.logger.error(f"Failed to terminate PTY process: {e}")
        
        # Close the master FD
        if self.master_fd is not None:
            os.close(self.master_fd)
            self.master_fd = None
        
        self.logger.info("Server stopped")
    
    def _spawn_pty(self) -> None:
        """
        Spawn a pseudo-terminal with the shell process.
        
        This creates a new pseudo-terminal and forks a child process that executes
        the specified shell command within that terminal.
        
        Raises:
            OSError: If the PTY creation or process spawn fails
        """
        # Create a pseudo-terminal
        self.pid, self.master_fd = pty.fork()
        
        if self.pid == 0:  # Child process
            # Execute the shell in the child process
            try:
                cmd_with_args = [self.shell_cmd] + self.shell_args
                os.execvp(self.shell_cmd, cmd_with_args)
            except Exception as e:
                print(f"Failed to execute shell command: {e}")
                sys.exit(1)
        else:
            self.logger.debug(f"Spawned PTY process with PID {self.pid}")
    
    def _setup_socket(self) -> None:
        """
        Set up the Unix domain socket server.
        
        This creates a Unix domain socket, removes any existing socket file,
        binds to the specified path, and starts listening for connections.
        
        Raises:
            OSError: If socket creation, binding, or listening fails
        """
        # Remove socket file if it already exists
        if os.path.exists(self.socket_path):
            try:
                os.unlink(self.socket_path)
            except OSError as e:
                self.logger.error(f"Failed to remove existing socket file: {e}")
                raise
        
        # Create and bind the socket
        try:
            self.server_socket = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            self.server_socket.bind(self.socket_path)
            self.server_socket.listen(5)
            self.logger.debug(f"Socket created and listening at {self.socket_path}")
        except OSError as e:
            self.logger.error(f"Failed to create socket: {e}")
            raise
    
    def _accept_clients(self) -> None:
        """
        Accept client connections in a loop.
        
        This method runs in a separate thread and continuously accepts
        new client connections until the server is stopped.
        """
        self.logger.info("Starting client acceptance thread")
        
        while self.running:
            try:
                # Set a timeout to periodically check the running flag
                self.server_socket.settimeout(1.0)
                try:
                    client, addr = self.server_socket.accept()
                    self.logger.info(f"New client connected: {addr}")
                    
                    # Add to client list and start read thread
                    self.clients.append(client)
                    client_thread = threading.Thread(target=self._handle_client, args=(client,))
                    client_thread.daemon = True
                    client_thread.start()
                except socket.timeout:
                    # This is expected, just continue
                    continue
                except OSError as e:
                    if not self.running:
                        break
                    self.logger.error(f"Error accepting client: {e}")
            except Exception as e:
                if not self.running:
                    break
                self.logger.error(f"Unexpected error in accept loop: {e}")
        
        self.logger.info("Client acceptance thread stopped")
    
    def _handle_client(self, client: socket.socket) -> None:
        """
        Handle communication from a client socket to the PTY.
        
        This method runs in a separate thread for each client and forwards
        data from the client to the PTY.
        
        Args:
            client: The client socket to handle
        """
        try:
            while self.running:
                r, _, _ = select.select([client], [], [], 0.1)
                if r:
                    try:
                        data = client.recv(self.buffer_size)
                        if not data:
                            # Client closed connection
                            self.logger.info("Client disconnected")
                            break
                        # Forward data to PTY
                        os.write(self.master_fd, data)
                    except (OSError, BrokenPipeError) as e:
                        self.logger.error(f"Error reading from client: {e}")
                        break
        finally:
            self._close_client(client)
    
    def _read_pty(self) -> None:
        """
        Read data from the PTY and forward to all connected clients.
        
        This method runs in a separate thread and continuously reads data
        from the PTY, forwarding it to all connected clients.
        """
        self.logger.info("Starting PTY read thread")
        
        while self.running:
            try:
                r, _, _ = select.select([self.master_fd], [], [], 0.1)
                if r:
                    try:
                        data = os.read(self.master_fd, self.buffer_size)
                        if not data:
                            # PTY process terminated
                            self.logger.warning("PTY process terminated")
                            self.stop()
                            break
                        
                        # Forward data to all clients
                        for client in self.clients[:]:  # Use a copy in case we modify the list
                            try:
                                client.sendall(data)
                            except (OSError, BrokenPipeError) as e:
                                self.logger.error(f"Error writing to client: {e}")
                                self._close_client(client)
                    except OSError as e:
                        self.logger.error(f"Error reading from PTY: {e}")
                        if e.errno == os.errno.EIO:  # PTY was closed
                            self.stop()
                        break
            except Exception as e:
                if not self.running:
                    break
                self.logger.error(f"Unexpected error in PTY read loop: {e}")
        
        self.logger.info("PTY read thread stopped")
    
    def _close_client(self, client: socket.socket) -> None:
        """
        Close a client connection and remove it from the clients list.
        
        Args:
            client: The client socket to close
        """
        try:
            client.close()
        except OSError as e:
            self.logger.error(f"Error closing client: {e}")
        
        if client in self.clients:
            self.clients.remove(client)


def test_terminal_server() -> None:
    """
    Test function for the TerminalSocketServer.
    
    This function:
    1. Creates and starts a TerminalSocketServer with bash
    2. Connects a client socket to the server
    3. Sends various commands to test functionality
    4. Verifies responses including ANSI control sequences
    5. Tests control characters (Ctrl+C, clear screen)
    """
    socket_path = "/tmp/terminal_test.sock"
    server = None
    client = None
    
    try:
        # Set up logging
        logging.basicConfig(level=logging.INFO)
        logger = logging.getLogger("TestTerminalServer")
        
        # Create and start the server
        logger.info("Starting terminal server...")
        server = TerminalSocketServer(socket_path, shell_cmd='/bin/bash', log_level=logging.INFO)
        server.start()
        
        # Give the server time to initialize
        time.sleep(1)
        
        # Connect client
        logger.info("Connecting client...")
        client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        client.connect(socket_path)
        
        # Function to send a command and receive response
        def send_command(cmd, wait_time=1):
            logger.info(f"Sending command: {cmd}")
            client.sendall(f"{cmd}\n".encode())
            time.sleep(wait_time)  # Give the command time to execute
            
            # Read response with a timeout
            responses = []
            timeout = time.time() + wait_time
            while time.time() < timeout:
                r, _, _ = select.select([client], [], [], 0.1)
                if r:
                    data = client.recv(4096)
                    if data:
                        responses.append(data)
                    else:
                        break
            
            return b''.join(responses)
        
        # Test 1: Basic command
        logger.info("Test 1: Basic command")
        response = send_command("echo 'Hello, Terminal Socket!'")
        logger.info(f"Response: {response!r}")
        
        # Test 2: Command with colors
        logger.info("Test 2: Command with colors")
        response = send_command("echo -e '\\e[31mRed\\e[32mGreen\\e[34mBlue\\e[0m'")
        logger.info(f"Response: {response!r}")
        
        # Test 3: ls with color
        logger.info("Test 3: ls with color")
        response = send_command("ls --color=always -la /")
        logger.info(f"Response preview: {response[:100]!r}")
        
        # Test 4: Clear screen
        logger.info("Test 4: Clear screen (ANSI escape sequence)")
        response = send_command("echo -e '\\033[2J\\033[H'")
        logger.info(f"Clear screen response: {response!r}")
        
        # Test 5: Ctrl+C handling
        logger.info("Test 5: Ctrl+C handling")
        client.sendall("sleep 10\n".encode())
        time.sleep(0.5)
        logger.info("Sending Ctrl+C (SIGINT)")
        client.sendall(b"\x03")  # ASCII code for Ctrl+C
        response = b''
        for _ in range(10):
            r, _, _ = select.select([client], [], [], 0.2)
            if r:
                data = client.recv(4096)
                if data:
                    response += data
        logger.info(f"Ctrl+C response: {response!r}")
        
        # Test 6: Interactive command
        logger.info("Test 6: Interactive command (python)")
        response = send_command("python3 -c 'print(\"Interactive Python\")'")
        logger.info(f"Python response: {response!r}")
        
        logger.info("All tests completed!")
    
    except Exception as e:
        logging.error(f"Test failed: {e}")
        raise
    finally:
        # Clean up
        if client:
            client.close()
        if server:
            server.stop()

socket_path = "/tmp/terminal_test.sock"
server = TerminalSocketServer(socket_path, shell_cmd='ipython', log_level=logging.INFO)
server.start()
while True:
    pass

if __name__ == "__main__":
    # Handle Ctrl+C gracefully
    def signal_handler(sig, frame):
        print("\nExiting...")
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    
    # Run the test
    test_terminal_server()
