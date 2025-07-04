#!/usr/bin/env python3
import os
import sys
import pdb
import socket

SOCK_PATH = '/tmp/pdb.sock'

class CustomPdb(pdb.Pdb):
    def __init__(self, completekey=None, stdin=None, stdout=None, **kwargs):
        super().__init__(completekey=completekey, stdin=stdin, stdout=stdout, **kwargs)
        self.prompt = '(custom-pdb) '

def setup_socket(path):
    # 如果已经存在旧的 socket 文件，先删除
    if os.path.exists(path):
        os.unlink(path)
    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server.bind(path)
    server.listen(1)
    print(f"[main] Waiting for debugger client to connect on {path}")
    conn, _ = server.accept()
    print("[main] Debugger client connected")
    server.close()
    return conn

def get_param(rank=0):
    # 1. 在 /tmp/pdb.sock 上等待客户端连接
    conn = setup_socket(SOCK_PATH+'.'+str(rank))
    print(f"[get_param] Connection established on {SOCK_PATH}.{rank}")

    # 2. 将 socket 包装成文本文件对象，分别作为 stdin/stdout
    conn_r = conn.makefile('r')
    conn_w = conn.makefile('w')

    # 3. 创建 Pdb，输入输出都在该 socket 上
    debugger = dict(
        # completekey=None,
        stdin=conn_r,
        stdout=conn_w
    )
    return debugger
    
def main():
    # 1. 在 /tmp/pdb.sock 上等待客户端连接
    conn = setup_socket(SOCK_PATH)

    # 2. 将 socket 包装成文本文件对象，分别作为 stdin/stdout
    conn_r = conn.makefile('r')
    conn_w = conn.makefile('w')

    # 3. 创建 Pdb，输入输出都在该 socket 上
    debugger = CustomPdb(
        completekey=None,
        stdin=conn_r,
        stdout=conn_w
    )

    # 4. 演示业务逻辑
    x = 10
    y = 20
    z = x + y

    # 5. 触发断点，所有交互都走 socket
    debugger.set_trace()

    # 6. 继续运行
    print(f"{x} + {y} = {z}")

    # 清理
    conn.close()
    if os.path.exists(SOCK_PATH):
        os.unlink(SOCK_PATH)

if __name__ == '__main__':
    main()
