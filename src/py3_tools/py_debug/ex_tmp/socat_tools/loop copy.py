import subprocess
import threading
import os
import sys
import time
import contextlib
import fcntl
import socket
import signal
import random
import termios

class StreamRedirector(contextlib.ContextDecorator):
    """Context manager for redirecting stdin/stdout/stderr to pipes for socat"""
    
    def __init__(self, port=7777, max_retries=5):
        self.initial_port = port
        self.port = port
        self.max_retries = max_retries
        self.pipe_dir = "/tmp"
        self.in_pipe = f"{self.pipe_dir}/py_stdin_{os.getpid()}.pipe"
        self.out_pipe = f"{self.pipe_dir}/py_stdout_{os.getpid()}.pipe"
        self.original_stdin = None
        self.original_stdout = None
        self.original_stderr = None
        self.process = None
        self.stdin_pipe = None
        self.stdout_pipe = None
        self.original_term_settings = None
            
    def __enter__(self):
        # Find an available port
        self.port = self.find_available_port()
        
        # Remove old pipes if they exist
        for pipe in [self.in_pipe, self.out_pipe]:
            if os.path.exists(pipe):
                os.remove(pipe)
        
        # Create the pipes
        for pipe in [self.in_pipe, self.out_pipe]:
            os.mkfifo(pipe)
        
        # Save original streams for restoration
        self.original_stdin = sys.stdin
        self.original_stdout = sys.stdout
        self.original_stderr = sys.stderr
        
        print(f"Starting socat server on port {self.port}", file=self.original_stdout)
        print(f"Connect with: socat -,raw,echo=0 TCP:localhost:{self.port}", file=self.original_stdout)
        
        # Use PTY for proper terminal handling instead of simple pipes
        cmd = f"socat TCP-LISTEN:{self.port},reuseaddr,fork,mode=777 " \
              f"EXEC:'bash -c \"stty raw -echo; cat {self.out_pipe} & cat > {self.in_pipe}\"',pty,rawer"
        
        # Run the server in the background
        self.process = subprocess.Popen(cmd, shell=True)
        
        # Give socat a moment to start
        time.sleep(0.5)
        
        # Check if process started successfully
        if self.process.poll() is not None:
            raise RuntimeError(f"Failed to start socat server on port {self.port}")
        
        # Open the pipes and redirect - now using non-blocking mode for stdin pipe
        fd_in = os.open(self.in_pipe, os.O_RDONLY | os.O_NONBLOCK)
        self.stdin_pipe = os.fdopen(fd_in, 'r')
        self.stdout_pipe = open(self.out_pipe, 'w', buffering=1)  # Line buffering
        
        # Set stdin back to blocking mode after it's open
        flags = fcntl.fcntl(fd_in, fcntl.F_GETFL)
        fcntl.fcntl(fd_in, fcntl.F_SETFL, flags & ~os.O_NONBLOCK)
        
        sys.stdin = self.stdin_pipe
        sys.stdout = self.stdout_pipe
        sys.stderr = self.stdout_pipe  # Redirect stderr to the same pipe as stdout
        
        return self

    def is_port_in_use(self, port):
        """Check if a port is already in use"""
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(('localhost', port))
                return False
            except socket.error:
                return True
    
    def find_available_port(self):
        """Find an available port, starting with self.initial_port"""
        if not self.is_port_in_use(self.initial_port):
            return self.initial_port
            
        for _ in range(self.max_retries):
            # Try a random port in the dynamic/private port range
            test_port = random.randint(49152, 65535)
            if not self.is_port_in_use(test_port):
                return test_port
                
        raise RuntimeError(f"Could not find an available port after {self.max_retries} attempts")
     
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        # Restore original streams
        sys.stdin = self.original_stdin
        sys.stdout = self.original_stdout
        sys.stderr = self.original_stderr
        
        # Close pipes
        if self.stdin_pipe:
            self.stdin_pipe.close()
        if self.stdout_pipe:
            self.stdout_pipe.close()
            
        # Remove the pipes
        for pipe in [self.in_pipe, self.out_pipe]:
            if os.path.exists(pipe):
                os.remove(pipe)
        
        # Terminate socat process
        if self.process:
            try:
                self.process.terminate()
                self.process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                # If termination takes too long, force kill
                self.process.kill()
        
        print(f"Socat server on port {self.port} has been shut down.", file=self.original_stdout)
        return False  # Let exceptions propagate

def cleanup_handler(signum, frame):
    """Signal handler to ensure clean exit on SIGINT/SIGTERM"""
    print("\nReceived termination signal. Cleaning up...", file=sys.__stdout__)
    sys.exit(0)

def run_interactive_loop():
    # Set signal handlers for graceful shutdown
    signal.signal(signal.SIGINT, cleanup_handler)
    signal.signal(signal.SIGTERM, cleanup_handler)
    
    # Start with redirected streams
    with StreamRedirector(port=7777):
        print("Stream redirection server started. Entering main loop...")
        
        # Main interactive loop
        try:
            while True:
                import ipdb
                ipdb.set_trace()
                # x = input("Enter something: ")
                # print(f"You entered: {x}")
                a = 1
                print(f"Debugging variable a: {a}")
        except KeyboardInterrupt:
            print("\nExiting interactive loop...")

if __name__ == "__main__":
    run_interactive_loop()