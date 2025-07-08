import subprocess
import os
import pty
import sys
import socket

def redirect_stdio_to_socat(socket_path="/tmp/my_socat_socket"):
    """Redirects stdin, stdout, and stderr to a socat TTY stream using Unix domain socket."""

    # 创建一个伪终端
    master_fd, slave_fd = pty.openpty()
    tty_name = os.ttyname(slave_fd)

    # 将标准输入、输出和错误重定向到伪终端
    # sys.stdin.close()
    # sys.stdout.close()
    # sys.stderr.close()


    sys.stdin = open(os.ttyname(slave_fd), 'r')
    sys.stdout = open(os.ttyname(slave_fd), 'w')
    sys.stderr = open(os.ttyname(slave_fd), 'w')

    sys.__stdin__ = sys.stdin
    sys.__stdout__ = sys.stdout
    sys.__stderr__ = sys.stderr

    # 创建一个 Unix domain socket
    # 确保 socket 文件不存在
    if os.path.exists(socket_path):
        os.remove(socket_path)

    # socat 命令，创建 Unix domain socket 服务器并将其连接到 TTY
    server_cmd = [
        'socat',
        f'UNIX-LISTEN:{socket_path},fork,reuseaddr',
        f'FILE:{tty_name},raw,echo=0'
    ]

    # 启动 socat 服务器进程
    print(f"Starting socat server with command: {' '.join(server_cmd)}")
    server_process = subprocess.Popen(server_cmd)

    print(f"socat server listening on Unix socket {socket_path}")
    print(f"Connect to it using: `socat PTY,link=/tmp/my_virtual_tty,raw,echo=0 UNIX:{socket_path}` in another terminal.")

    return server_process, socket_path

if __name__ == "__main__":
    socket_path = "/tmp/my_socat_socket"  # 定义 Unix socket 的路径
    server, socket_path = redirect_stdio_to_socat(socket_path)

    try:
        # 你的程序逻辑在这里
        while True:
            line = input("Enter something: ")
            print(f"You entered: {line}")
    except KeyboardInterrupt:
        print("Interrupted, cleaning up...")
        server.terminate()
        server.wait()