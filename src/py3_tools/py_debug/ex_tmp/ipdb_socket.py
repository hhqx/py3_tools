#!/usr/bin/env python3
import os
import socket
import sys
from ipdb import Pdb

SOCK_PATH = '/tmp/ipdb.sock'

def setup_socket(path):
    # 如果旧的 socket 文件存在，先删除
    if os.path.exists(path):
        os.unlink(path)
    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server.bind(path)
    server.listen(1)
    print(f"[main] Waiting for ipdb client to connect on {path}")
    conn, _ = server.accept()
    print("[main] ipdb client connected")
    server.close()
    return conn

class CustomIpdb(Pdb):
    def __init__(self, completekey=None, stdin=None, stdout=None, **kwargs):
        # 直接调用 ipdb.Pdb，传入 socket 流
        super().__init__(completekey=completekey, stdin=stdin, stdout=stdout, **kwargs)
        self.prompt = '(custom-ipdb) '

def main():
    # 1. 启动 socket 服务，等待客户端连接
    conn = setup_socket(SOCK_PATH)
    conn_r = conn.makefile('r')
    conn_w = conn.makefile('w')

    # 2. 准备一些示例变量
    x = 10
    y = 20
    z = x + y

    # 3. 创建带 stdin/stdout 重定向的 ipdb 调试器
    dbg = CustomIpdb(
        completekey=None,
        stdin=conn_r,
        stdout=conn_w
    )
    dbg.use_rawinput = True     # ← 关键：启用原始输入，否则 ipdb 会拒绝
    dbg.set_trace()

    # 4. 在这里打断点，所有调试交互都会走 socket
    dbg.set_trace()

    # 5. 继续执行
    print(f"{x} + {y} = {z}")

    # 6. 清理
    conn.close()
    if os.path.exists(SOCK_PATH):
        os.unlink(SOCK_PATH)

if __name__ == '__main__':
    main()
