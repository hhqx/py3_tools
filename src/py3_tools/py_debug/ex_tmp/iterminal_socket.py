#!/usr/bin/env python3
import os
import socket
import sys

# 假设 TerminalPdb 定义在同目录下的 termpdb.py 中，
# 或者你可以直接把上面那段 TerminalPdb 类粘到这里。
from IPython.terminal.debugger import TerminalPdb  

SOCK_PATH = '/tmp/ipdb_term.sock'

def setup_socket(path):
    # 如果旧的 socket 文件存在，先删掉
    if os.path.exists(path):
        os.unlink(path)
    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server.bind(path)
    server.listen(1)
    print(f"[main] Waiting for client on {path}", file=sys.stderr)
    conn, _ = server.accept()
    print(f"[main] Client connected", file=sys.stderr)
    server.close()
    return conn

def main():
    # 1. 启动 socket 并等待客户端连接
    conn = setup_socket(SOCK_PATH)

    # 2. 将 socket 封装成文件对象，用于 TerminalPdb 的 stdin/stdout
    conn_r = conn.makefile('r')
    conn_w = conn.makefile('w')

    # 3. 示例业务逻辑
    x, y = 10, 20
    z = x + y

    # 4. 创建调试器实例，重定向输入输出到 socket
    dbg = TerminalPdb(stdin=conn_r, stdout=conn_w)
    dbg.use_rawinput = True     # ← 关键：启用原始输入，否则 ipdb 会拒绝
    # dbg.set_trace()
    dbg.set_trace()  # 在这里进入调试，所有交互都走 socket

    # 5. 调试结束后继续执行
    print(f"{x} + {y} = {z}")

    # 6. 清理
    conn.close()
    if os.path.exists(SOCK_PATH):
        os.unlink(SOCK_PATH)

if __name__ == '__main__':
    main()
