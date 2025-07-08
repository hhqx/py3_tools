import pty
import os

def read_from_master(fd):
    data = os.read(fd, 1024)
    # print("Wizard says:", data)
    # print("Wizard says:", data.decode())
    return "abcd\r\n".encode()  # Return a string to simulate reading from master

# pty.spawn(["ls", "-all"], master_read=read_from_master)
pty.spawn(["echo", 'halo'], master_read=read_from_master)