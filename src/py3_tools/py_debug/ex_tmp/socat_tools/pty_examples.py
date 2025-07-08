#!/usr/bin/env python3
# filepath: /home/hqx/myprojects/py_tools/socat_tools/pty_examples.py
import os
import pty
import select
import sys
import time
import re
import socket
import termios
import tty
import signal

"""
PTY.spawn() 实用示例集合
每个函数展示 pty.spawn() 的不同用法场景
"""

def example1_basic():
    """基本用法：简单运行命令并捕获输出"""
    print("\r\n=== 示例1：基本用法 ===")
    
    def read_from_master(fd):
        data = os.read(fd, 1024)
        print(f"捕获到 {len(data)} 字节输出")
        return data
        
    print("运行 'ls -la' 命令...")
    pty.spawn(["ls", "-la"], master_read=read_from_master)
    print("命令已完成")


def example2_terminal_recorder():
    """终端录制：记录终端会话到文件"""
    print("\r\n=== 示例2：终端会话录制 ===")
    
    # 创建记录文件
    filename = "/tmp/terminal_recording.txt"
    with open(filename, 'wb') as script:
        # 写入起始标记
        script.write(f"=== 会话开始于 {time.asctime()} ===\r\n\r\n".encode())
        
        def read_and_record(fd):
            data = os.read(fd, 1024)
            script.write(data)
            return data
        
        print(f"开始记录会话，输入 'exit' 结束...")
        pty.spawn([os.environ.get('SHELL', '/bin/bash')], read_and_record)
        
        # 写入结束标记
        script.write(f"\r\n\r\n=== 会话结束于 {time.asctime()} ===\r\n".encode())
    
    print(f"会话已记录到 {filename}")


def example3_output_filter():
    """输出过滤器：修改程序输出"""
    print("\r\n=== 示例3：输出过滤器 ===")
    
    def filter_output(fd):
        data = os.read(fd, 1024)
        if data:
            # 转换为文本处理
            text = data.decode('utf-8', errors='replace')
            
            # 添加行号和时间戳
            lines = text.splitlines()
            result = []
            for i, line in enumerate(lines):
                timestamp = time.strftime('%H:%M:%S')
                result.append(f"[{timestamp} #{i+1}] {line}")
            
            # 如果原始数据有尾随换行，保留它
            if text.endswith('\r\n'):
                modified = '\r\n'.join(result) + '\r\n'
            else:
                modified = '\r\n'.join(result)
            
            # 转回字节
            return modified.encode('utf-8')
        return data
    
    print("运行 'ls -la' 命令并为每行添加时间戳和行号...")
    pty.spawn(["ls", "-la"], master_read=filter_output)


def example4_interactive_automation():
    """交互式程序自动化：与需要输入的程序交互"""
    print("\r\n=== 示例4：交互式程序自动化 ===")
    
    # 模拟交互程序的脚本
    script_content = """#!/bin/bash
echo "请输入您的名字："
read name
echo "您好，$name！"
echo "请输入密码："
read -s password
echo "密码已接收"
echo "请选择一个选项 (1-3)："
read option
echo "您选择了选项 $option"
echo "完成！"
"""
    
    # 创建临时脚本文件
    script_path = "/tmp/interactive_script.sh"
    with open(script_path, 'w') as f:
        f.write(script_content)
    os.chmod(script_path, 0o750)
    
    # 准备自动响应
    responses = [
        ("请输入您的名字", "Python用户\r\n"),
        ("请输入密码", "秘密密码\r\n"),
        ("请选择一个选项", "2\r\n")
    ]
    
    current_response = 0
    
    def automated_read(fd):
        nonlocal current_response
        data = os.read(fd, 1024)
        print("程序输出:", data.decode('utf-8', errors='replace').strip())
        return data
    
    def automated_input(fd):
        nonlocal current_response
        # 检查是否所有响应都已发送
        if current_response >= len(responses):
            time.sleep(0.1)  # 等待最后的输出
            return b""  # 返回空表示结束
            
        # 从程序读取一行输入，检查是否匹配预期提示
        prompt = responses[current_response][0]
        response = responses[current_response][1]
        
        # 假设我们已经看到了提示
        print(f"检测到提示 '{prompt}', 发送响应: '{response.strip()}'")
        current_response += 1
        return response.encode('utf-8')
    
    print("自动化运行交互式脚本...")
    pty.spawn([script_path], master_read=automated_read, stdin_read=automated_input)
    print("自动化完成")
    os.unlink(script_path)


def example5_terminal_proxy():
    """终端代理：通过网络转发终端会话"""
    print("\r\n=== 示例5：终端网络代理 ===")
    
    try:
        # 创建本地服务器
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        port = 8899
        server.bind(('127.0.0.1', port))
        server.listen(1)
        
        print(f"启动终端代理服务器在端口 {port}")
        print(f"请打开另一个终端并运行: nc 127.0.0.1 {port}")
        print("等待连接...")
        
        client, addr = server.accept()
        print(f"客户端已连接: {addr}")
        
        # 保存原始终端设置
        old_settings = termios.tcgetattr(sys.stdin.fileno())
        
        # 设置非阻塞模式
        client.setblocking(False)
        
        def proxy_read(fd):
            try:
                # 检查socket是否有数据
                r, _, _ = select.select([client], [], [], 0)
                if r:
                    client_data = client.recv(1024)
                    if client_data:
                        os.write(fd, client_data)  # 转发到PTY
            except:
                pass
            
            # 从PTY读取
            data = os.read(fd, 1024)
            if data:
                try:
                    client.sendall(data)  # 转发到客户端
                except:
                    pass
            return data
        
        # 运行shell并代理所有I/O
        shell = os.environ.get('SHELL', '/bin/bash')
        try:
            # 将TERM环境变量传递给子进程
            pty.spawn([shell], master_read=proxy_read)
        except (EOFError, OSError):
            pass
        finally:
            # 恢复终端设置
            termios.tcsetattr(sys.stdin.fileno(), termios.TCSADRAIN, old_settings)
            client.close()
            server.close()
            print("终端代理会话已结束")
    
    except Exception as e:
        print(f"错误: {e}")
        return


def example6_terminal_multiplexer():
    """终端复用器：允许多个程序共享同一个终端会话"""
    print("\r\n=== 示例6：终端复用器 ===")
    
    # 创建master/slave对
    master1, slave1 = pty.openpty()
    master2, slave2 = pty.openpty()
    
    # 创建共享终端
    def terminal_multiplexer():
        print("启动终端复用器 (按 Ctrl+D 两次退出)")
        print("同时运行: 'ls -la' 和 'ps aux | grep python'")
        
        # 创建子进程
        pid1 = os.fork()
        if pid1 == 0:  # 第一个子进程
            os.close(master1)
            os.dup2(slave1, 0)  # stdin
            os.dup2(slave1, 1)  # stdout
            os.dup2(slave1, 2)  # stderr
            os.execvp("ls", ["ls", "-la"])
            os._exit(0)
        
        os.close(slave1)
        
        pid2 = os.fork()
        if pid2 == 0:  # 第二个子进程
            os.close(master2)
            os.dup2(slave2, 0)  # stdin
            os.dup2(slave2, 1)  # stdout
            os.dup2(slave2, 2)  # stderr
            os.execvp("sh", ["sh", "-c", "ps aux | grep python"])
            os._exit(0)
        
        os.close(slave2)
        
        # 主循环：从两个终端读取输出，并显示给用户
        try:
            while True:
                # 等待任一终端有数据可读
                r, _, _ = select.select([master1, master2], [], [])
                
                for fd in r:
                    try:
                        data = os.read(fd, 1024)
                        if not data:
                            continue
                            
                        # 给不同终端添加颜色标记
                        if fd == master1:
                            sys.stdout.write("\033[32m")  # 绿色
                            sys.stdout.write("[程序1] ")
                        else:
                            sys.stdout.write("\033[33m")  # 黄色
                            sys.stdout.write("[程序2] ")
                            
                        sys.stdout.write("\033[0m")  # 重置颜色
                        sys.stdout.buffer.write(data)
                        sys.stdout.flush()
                    except OSError:
                        pass
        except KeyboardInterrupt:
            pass
        finally:
            # 清理
            os.close(master1)
            os.close(master2)
            try:
                os.waitpid(pid1, 0)
                os.waitpid(pid2, 0)
            except:
                pass
                
    terminal_multiplexer()


def example7_shell_session_logger():
    """Shell会话详细记录器：记录时间戳和执行状态"""
    print("\r\n=== 示例7：详细Shell会话记录器 ===")
    
    log_file = "/tmp/detailed_session.log"
    
    with open(log_file, 'w') as log:
        log.write(f"=== 会话开始于 {time.ctime()} ===\r\n\r\n")
        
        command_buffer = ""
        last_prompt = ""
        in_command = False
        command_start_time = None
        
        def logger_read(fd):
            nonlocal command_buffer, last_prompt, in_command, command_start_time
            
            data = os.read(fd, 1024)
            if not data:
                return data
                
            output = data.decode('utf-8', errors='replace')
            
            # 检测提示符和命令
            if not in_command and re.search(r'[$#>] $', output):
                last_prompt = output
                in_command = True
                
            # 如果我们在命令输入模式，检查回车键
            elif in_command and '\r' in output:
                in_command = False
                if command_buffer.strip():
                    log.write(f"[{time.ctime()}] 命令: {command_buffer.strip()}\r\n")
                    command_start_time = time.time()
                command_buffer = ""
            
            return data
            
        def logger_input(fd):
            nonlocal command_buffer, in_command
            
            data = os.read(fd, 1024)
            if not data:
                return data
                
            # 跟踪命令输入
            if in_command:
                try:
                    command_buffer += data.decode('utf-8', errors='replace')
                except:
                    pass
                    
            return data
        
        print(f"启动详细Shell会话记录，日志保存到: {log_file}")
        print("输入 'exit' 结束记录")
        
        # 运行shell并记录
        pty.spawn([os.environ.get('SHELL', '/bin/bash')], 
                  master_read=logger_read,
                  stdin_read=logger_input)
        
        log.write(f"\r\n=== 会话结束于 {time.ctime()} ===\r\n")
        
    print(f"记录完成，日志保存到: {log_file}")


if __name__ == "__main__":
    # 运行所有示例
    try:
        # 为每个例子设置简单的超时保护
        def timeout_handler(signum, frame):
            raise TimeoutError("示例执行超时")
            
        # 用户可以选择运行特定示例
        if len(sys.argv) > 1:
            example_num = int(sys.argv[1])
            examples = {
                1: example1_basic,
                2: example2_terminal_recorder,
                3: example3_output_filter,
                4: example4_interactive_automation, 
                5: example5_terminal_proxy,
                6: example6_terminal_multiplexer,
                7: example7_shell_session_logger
            }
            
            if example_num in examples:
                print(f"运行示例 {example_num}")
                examples[example_num]()
            else:
                print(f"示例 {example_num} 不存在。可用示例: 1-7")
        else:
            # 默认按顺序运行所有简单示例
            example1_basic()
            example3_output_filter()
            example4_interactive_automation()
            
            print("\r\n其它高级示例可用命令行参数运行:")
            print("  python pty_examples.py 2    # 终端会话录制")
            print("  python pty_examples.py 5    # 终端网络代理")
            print("  python pty_examples.py 6    # 终端复用器")
            print("  python pty_examples.py 7    # 详细Shell会话记录器")
            
    except KeyboardInterrupt:
        print("\r\n用户中断执行")
    except Exception as e:
        print(f"执行错误: {e}")