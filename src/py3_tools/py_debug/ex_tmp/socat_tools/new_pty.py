import os
import sys
import time
import fcntl
import subprocess
import atexit
import tempfile
import signal

class PtyRedirect:
    """
    Redirects stdin/stdout/stderr through a Unix socket using socat.
    Provides a pseudo-terminal interface that remote terminals can connect to.
    """
    def __init__(self, socket_path=None, pty_mode=True):
        """
        Initialize the redirector.
        
        Args:
            socket_path: Path to the Unix socket (default: temporary file)
            pty_mode: Whether to use PTY mode (True) or pipe mode (False)
        """
        # Save original stdin/stdout/stderr
        self.original_stdin = sys.stdin
        self.original_stdout = sys.stdout
        self.original_stderr = sys.stderr
        
        # Create a unique socket path if not provided
        if socket_path is None:
            # socket_dir = tempfile.mkdtemp()
            socket_path = os.path.join('/tmp/', 'pty.sock')
        if os.path.exists(socket_path):
            os.remove(socket_path)
        
        self.socket_path = socket_path
        
        # Create unique pipe names
        self.in_pipe = f"{socket_path}.in"
        self.out_pipe = f"{socket_path}.out"
        
        # Initialize process and pipe variables
        self.process = None
        self.stdin_pipe = None
        self.stdout_pipe = None
        
        # Mode selection
        self.pty_mode = pty_mode
        
        # Register cleanup on exit
        atexit.register(self.cleanup)
    
    def start(self):
        """Start the redirection."""
        if self.pty_mode:
            self._setup_pty()
        else:
            self._setup_pipes()
            
        print(f"Redirection active. Socket: {self.socket_path}", file=self.original_stdout)
    
    def _setup_pty(self):
        """Setup PTY mode"""
        # Create a PTY link
        pty_link = f"/tmp/jupyter_pty_{os.getpid()}"
        if os.path.exists(pty_link):
            os.unlink(pty_link)
        
        print(f"Starting socat server on socket {self.socket_path}", file=self.original_stdout)
        print(f"Connect with: socat -d -d file:`tty`,raw,echo=0 UNIX-CONNECT:{self.socket_path}", file=self.original_stdout)

        # Start socat to connect UNIX socket to PTY
        cmd = f"socat -d -d UNIX-LISTEN:{self.socket_path},fork,reuseaddr PTY,link={pty_link},raw,echo=0"
        
        # Run the server in the background
        self.process = subprocess.Popen(cmd, shell=True, stderr=subprocess.PIPE)
        
        # Give socat a moment to start
        time.sleep(0.5)
        
        # Check if process started successfully
        if self.process.poll() is not None:
            raise RuntimeError(f"Failed to start socat server on socket {self.socket_path}")
        
        # Open the PTY link
        self.pty_fd = os.open(pty_link, os.O_RDWR | os.O_NONBLOCK)
        
        # Set stdin/stdout/stderr to the PTY
        sys.stdin = os.fdopen(self.pty_fd, 'r')
        sys.stdout = os.fdopen(self.pty_fd, 'w')
        sys.stderr = sys.stdout
        
    
    def _setup_pipes(self):
        """Setup regular pipes (non-TTY mode)"""
        # Remove old pipes if they exist
        for pipe in [self.in_pipe, self.out_pipe]:
            if os.path.exists(pipe):
                os.remove(pipe)
        
        # Create the pipes
        for pipe in [self.in_pipe, self.out_pipe]:
            os.mkfifo(pipe)
            
        print(f"Starting socat server on socket {self.socket_path}", file=self.original_stdout)
        print(f"Connect with: socat -d -d file:`tty`,raw,echo=0 UNIX-CONNECT:{self.socket_path}", file=self.original_stdout)

        # Start socat to connect UNIX socket to our pipes
        cmd = f"socat -d -d UNIX-LISTEN:{self.socket_path},fork SYSTEM:'cat {self.out_pipe} & cat > {self.in_pipe}',pty"
        
        # Run the server in the background
        self.process = subprocess.Popen(cmd, shell=True)
        
        # Give socat a moment to start
        time.sleep(0.5)
        
        # Check if process started successfully
        if self.process.poll() is not None:
            raise RuntimeError(f"Failed to start socat server on socket {self.socket_path}")
        
        # Open the pipes and redirect - now using non-blocking mode for stdin pipe
        fd_in = os.open(self.in_pipe, os.O_RDONLY | os.O_NONBLOCK)
        self.stdin_pipe = os.fdopen(fd_in, 'r')
        self.stdout_pipe = open(self.out_pipe, 'w')
        
        # Set stdin back to blocking mode after it's open
        flags = fcntl.fcntl(fd_in, fcntl.F_GETFL)
        fcntl.fcntl(fd_in, fcntl.F_SETFL, flags & ~os.O_NONBLOCK)
        
        sys.stdin = self.stdin_pipe
        sys.stdout = self.stdout_pipe
        sys.stderr = self.stdout_pipe
    
    def restore(self):
        """Restore original stdin/stdout/stderr"""
        if sys.stdin is not self.original_stdin:
            sys.stdin = self.original_stdin
        if sys.stdout is not self.original_stdout:
            sys.stdout = self.original_stdout
        if sys.stderr is not self.original_stderr:
            sys.stderr = self.original_stderr
    
    def cleanup(self):
        """Clean up resources"""
        self.restore()
        
        # Close pipes
        if self.stdin_pipe:
            self.stdin_pipe.close()
        if self.stdout_pipe:
            self.stdout_pipe.close()
        
        # Close PTY if we were using it
        if hasattr(self, 'pty_fd') and self.pty_fd is not None:
            os.close(self.pty_fd)
            if os.path.exists("/tmp/ptylink"):
                os.unlink("/tmp/ptylink")
        
        # Remove named pipes
        for pipe in [self.in_pipe, self.out_pipe]:
            if os.path.exists(pipe):
                os.remove(pipe)
        
        # Terminate socat process
        if self.process:
            self.process.terminate()
            try:
                self.process.wait(timeout=1)
            except subprocess.TimeoutExpired:
                self.process.kill()
        
        # Remove socket file
        if os.path.exists(self.socket_path):
            os.remove(self.socket_path)

import time
# from pty_redirect import PtyRedirect

# Create and start the redirection with PTY mode
redirector = PtyRedirect(pty_mode=True)
redirector.start()

# Now your program can use stdin/stdout/stderr normally
# and remote terminals can connect to the socket

print("Hello from the redirected terminal!")
name = input("Enter your name: ")
print(f"Nice to meet you, ")

while True:
    a = input("input something: ")
    print(f"You entered: {a}")
    print(f"You entered: {a.encode('utf-8')}")
    if a == 'q':
        print("Exiting...")
        break
    
    

import ipdb

ipdb.set_trace()

# Run for a while accepting connections
try:
    while True:
        print("I'm still running...")
        time.sleep(5)
except KeyboardInterrupt:
    print("Shutting down...")

# Clean up when done
redirector.cleanup()