import socket
import sys
import os
import pty
import termios
import select
import threading
import time
import io

def remote_trace(host='localhost', port=5678):
    """
    Set up a remote ipdb debugging session with proper terminal support.
    
    When execution reaches this point, it will wait for a remote connection
    before starting the ipdb debugger.
    
    Args:
        host (str): Host to listen on (default: localhost)
        port (int): Port to listen on (default: 5678)
    
    Usage:
        1. Add this function to your code
        2. Insert `remote_trace()` at any point where you want to debug
        3. When code reaches this point, connect from another terminal:
           socat -,raw,echo=0 TCP:localhost:5678
    """
    # Create a server socket
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind((host, port))
    server.listen(1)
    
    print(f"[*] Remote ipdb waiting for connection at {host}:{port}")
    print(f"[*] Connect with: socat -,raw,echo=0 TCP:{host}:{port}")
    
    # Accept client connection
    client_socket, addr = server.accept()
    print(f"[*] Connection from {addr[0]}:{addr[1]}")
    
    # Create a pseudo-terminal
    master_fd, slave_fd = pty.openpty()
    
    # Save original stdin/stdout/stderr
    orig_stdin = sys.stdin
    orig_stdout = sys.stdout
    orig_stderr = sys.stderr
    
    # Create a custom TextIOWrapper with encoding
    class EncodedIO(io.TextIOWrapper):
        def __init__(self, fd, **kwargs):
            # Open the file descriptor in binary mode
            binary_io = os.fdopen(fd, 'rb+', buffering=0)
            # Initialize TextIOWrapper with encoding
            super().__init__(binary_io, encoding='utf-8', errors='replace', write_through=True, **kwargs)
    
    # Open slave fd with encoding
    slave_file = EncodedIO(slave_fd)
    
    # Create threads to relay data between socket and pty
    def socket_to_pty():
        while True:
            try:
                data = client_socket.recv(1024)
                if not data:
                    break
                os.write(master_fd, data)
            except Exception as e:
                print(f"Socket to PTY error: {e}")
                break
    
    def pty_to_socket():
        while True:
            try:
                r, _, _ = select.select([master_fd], [], [], 0.1)
                if master_fd in r:
                    data = os.read(master_fd, 1024)
                    if not data:
                        break
                    client_socket.sendall(data)
            except Exception as e:
                print(f"PTY to socket error: {e}")
                break
    
    # Start relay threads
    socket_thread = threading.Thread(target=socket_to_pty, daemon=True)
    pty_thread = threading.Thread(target=pty_to_socket, daemon=True)
    socket_thread.start()
    pty_thread.start()
    
    try:
        # Redirect stdin/stdout/stderr to the pty
        sys.stdin = slave_file
        sys.stdout = slave_file
        sys.stderr = slave_file
        
        # Initialize terminal settings for ipdb
        term_settings = termios.tcgetattr(slave_fd)
        term_settings[3] = term_settings[3] & ~termios.ECHO  # Disable echo
        termios.tcsetattr(slave_fd, termios.TCSANOW, term_settings)
        
        # Allow time for the terminal to initialize
        time.sleep(0.1)
        
        # Import ipdb here to avoid initialization issues
        try:
            import ipdb
            ipdb.set_trace(context=5)
        except Exception as e:
            print(f"Error starting ipdb: {e}")
            # Fallback to standard pdb if ipdb fails
            import pdb
            pdb.set_trace()
        
    finally:
        # Restore original stdin/stdout/stderr
        sys.stdin = orig_stdin
        sys.stdout = orig_stdout
        sys.stderr = orig_stderr
        
        # Clean up
        slave_file.close()
        os.close(master_fd)
        client_socket.close()
        server.close()

def example_function():
    a = 1
    b = 2
    remote_trace()  # Execution will pause here waiting for connection
    c = a + b
    return c

if __name__ == "__main__":
    result = example_function()
    print(f"Result: {result}")

'''
# Connect with this command for proper terminal support:
socat -,raw,echo=0 TCP:localhost:5678

# Alternative for better terminal handling:
socat $(tty),raw,echo=0 TCP:localhost:5678
'''