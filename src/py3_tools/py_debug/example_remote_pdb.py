"""
RemotePdb 使用指南

RemotePdb 是一个允许你远程调试 Python 程序的工具。
这对于在无法直接访问终端的环境中（例如容器、远程服务器）特别有用。

使用步骤:
1. 安装 remote_pdb 包: pip install remote-pdb
2. 在代码中需要调试的位置添加 RemotePdb('host', port).set_trace()
3. 当执行到该行时，程序会暂停并等待远程连接
4. 在另一个终端中使用 telnet 或 nc 连接到指定的主机和端口:
   telnet 127.0.0.1 4444
   或
   nc 127.0.0.1 4444

连接后，你将获得一个标准的 pdb 调试界面，可以使用常见的 pdb 命令:
- n: 执行下一行（step over）
- s: 步入函数（step into）
- c: 继续执行（continue）
- p 变量名: 打印变量值
- q: 退出调试器
- h: 查看帮助
"""

from remote_pdb import RemotePdb


def multiply(a, b):
    result = a * b
    # 在这里开启远程调试，监听所有接口 4444 端口
    RemotePdb('0.0.0.0', 4444).set_trace()
    return result


def calculate_factorial(n):
    """Calculate factorial of n recursively."""
    if n <= 1:
        return 1
    return n * calculate_factorial(n - 1)


def main():
    print("Starting remote PDB example...")
    print("Connect using: telnet 127.0.0.1 4444")

    # Set a breakpoint
    RemotePdb('127.0.0.1', 4444).set_trace()

    # This code will wait for debugger input
    n = 5
    result = calculate_factorial(n)
    print(f"Factorial of {n} is {result}")

    # Another breakpoint
    RemotePdb('127.0.0.1', 4444).set_trace()

    # More code to debug
    for i in range(3):
        print(f"Step {i}: {i * result}")

    print("Finished!")


if __name__ == "__main__":
    x = 7
    y = 6
    print(f"{x} * {y} =", multiply(x, y))
    main()
