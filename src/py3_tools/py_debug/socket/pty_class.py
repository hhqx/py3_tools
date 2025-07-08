import os
import pty
import termios
import select
import threading
import time
import io
import pdb
import sys
import logging
from typing import Optional, TextIO

# Configure logger
logger = logging.getLogger(__name__)

class PtyPdb(pdb.Pdb):
    """
    A PDB subclass that redirects I/O through a pseudo-terminal (PTY).
    
    This class creates a pseudo-terminal and redirects the debugger's
    input/output through it, allowing for better terminal handling and
    interaction with the debugger.
    
    Attributes:
        master_fd: Master file descriptor for the pseudo-terminal
        slave_fd: Slave file descriptor for the pseudo-terminal
        slave_file: File-like object for interacting with slave fd
        orig_stdin: Original stdin stream
        orig_stdout: Original stdout stream
        orig_stderr: Original stderr stream
    """
    
    def __init__(self, **kwargs):
        """
        Initialize a PtyPdb instance.
        
        Args:
            **kwargs: Additional keyword arguments to pass to pdb.Pdb constructor
        """
        # Create pseudo-terminal
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
        
        # Store original streams
        self.orig_stdin = None
        self.orig_stdout = None
        self.orig_stderr = None
        
        # Set up terminal settings
        self._setup_terminal()
        
        # Set a custom prompt
        kwargs['prompt'] = kwargs.get('prompt', '(pty-pdb) ')
        
        # Initialize the parent Pdb class with the pty as stdin/stdout
        super().__init__(stdin=self.slave_file, stdout=self.slave_file, **kwargs)

    def _setup_terminal(self) -> None:
        """Configure terminal settings for the pseudo-terminal."""
        try:
            # Initialize terminal settings
            term_settings = termios.tcgetattr(self.slave_fd)
            # Disable echo to prevent double echoing
            term_settings[3] = term_settings[3] & ~termios.ECHO
            termios.tcsetattr(self.slave_fd, termios.TCSANOW, term_settings)
        except (termios.error, OSError) as e:
            logger.warning(f"Failed to configure terminal settings: {e}")

    def redirect_streams(self) -> None:
        """Redirect sys.stdin/stdout/stderr to the pseudo-terminal."""
        # Save original streams
        self.orig_stdin = sys.stdin
        self.orig_stdout = sys.stdout
        self.orig_stderr = sys.stderr
        
        # Redirect to the pty
        sys.stdin = self.slave_file
        sys.stdout = self.slave_file
        sys.stderr = self.slave_file

    def restore_streams(self) -> None:
        """Restore original sys.stdin/stdout/stderr streams."""
        if self.orig_stdin:
            sys.stdin = self.orig_stdin
            self.orig_stdin = None
            
        if self.orig_stdout:
            sys.stdout = self.orig_stdout
            self.orig_stdout = None
            
        if self.orig_stderr:
            sys.stderr = self.orig_stderr
            self.orig_stderr = None

    def cleanup(self) -> None:
        """Clean up resources (file descriptors and streams)."""
        # Restore original streams
        self.restore_streams()
        
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

    def __enter__(self):
        """Support context manager protocol."""
        self.redirect_streams()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Clean up resources when exiting context manager."""
        self.cleanup()
        
    def do_quit(self, arg):
        """Handle the quit command to ensure proper cleanup."""
        self.cleanup()
        return super().do_quit(arg)
    
    do_q = do_quit  # alias
    
    def do_exit(self, arg):
        """Handle the exit command to ensure proper cleanup."""
        self.cleanup()
        return super().do_exit(arg)

    def get_master_fd(self) -> int:
        """Get the master file descriptor for external use."""
        return self.master_fd


def pty_set_trace(frame=None):
    """
    Start a PtyPdb debugging session through a pseudo-terminal.
    
    Args:
        frame: The frame to debug (default: caller's frame)
    
    Usage:
        1. Add this function to your code
        2. Call pty_set_trace() at any point where you want to debug
        3. The debugger will start with PTY support for better terminal handling
    """
    if frame is None:
        frame = sys._getframe().f_back
    
    # Create and start the debugger
    with PtyPdb() as debugger:
        debugger.set_trace(frame)


def pty_post_mortem(tb=None):
    """
    Start a post-mortem debugging session through a pseudo-terminal.
    
    Args:
        tb: Traceback object to debug (default: current exception)
        
    Usage:
        try:
            # Some code that might raise an exception
            result = 1/0
        except Exception:
            pty_post_mortem()
    """
    if tb is None:
        # Get current exception
        _, _, tb = sys.exc_info()
    
    if tb is None:
        raise ValueError("No traceback provided and no exception is being handled")
    
    # Create and start the debugger
    with PtyPdb() as debugger:
        debugger.reset()
        debugger.interaction(None, tb)


if __name__ == "__main__":
    def test_function():
        """Test function to demonstrate PTY debugging."""
        a = 1
        b = 2
        pty_set_trace()  # Debugger will start here
        c = a + b
        return c
    
    print("Testing PTY debugging...")
    result = test_function()
    print(f"Result: {result}")
