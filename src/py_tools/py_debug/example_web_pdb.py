"""
web_pdb 使用指南

web_pdb 是一个基于 Web 的 Python 调试器，提供了通过浏览器进行代码调试的功能。
这对于在没有图形界面或者远程环境中调试特别有用。

使用步骤:
1. 安装 web_pdb 包: pip install web-pdb
2. 在代码中需要调试的位置添加 set_trace()
3. 当执行到该行时，程序会暂停并启动一个 Web 服务器
4. 在浏览器中打开提示的 URL（默认为 http://localhost:5555）
5. 使用 Web 界面进行调试，界面包含:
   - 代码查看窗口
   - 变量查看窗口
   - 命令输入框

常用调试命令:
- n: 执行下一行（step over）
- s: 步入函数（step into）
- c: 继续执行（continue）
- p 变量名: 打印变量值
- q: 退出调试器
- h: 查看帮助

优势:
- 提供图形化界面，比传统的 pdb 更直观
- 可以远程访问，适合在服务器环境下使用
- 可以查看局部变量和全局变量的值
"""
from web_pdb import set_trace
import time
import random

class DataProcessor:
    """示例类，用于演示如何使用 web_pdb 进行调试."""

    def __init__(self, data):
        self.data = data
        self.processed = None

    def process(self):
        """处理数据，执行一些复杂的操作."""
        results = []

        # 在处理之前设置断点
        set_trace()

        # 处理每个项目
        for i, item in enumerate(self.data):
            # 一些任意的处理
            processed_item = item * i
            results.append(processed_item)

            # 模拟工作
            time.sleep(0.2)

        self.processed = results
        return results

def generate_sample_data(size=10):
    """生成一些示例数据."""
    return [random.randint(1, 100) for _ in range(size)]

def add(a, b):
    result = a + b
    # 在这里停下来，让你通过浏览器观察上下文
    set_trace()
    return result

def main():
    print("启动 Web PDB 示例...")
    print("请在浏览器中连接: http://localhost:5555")

    # 生成数据
    data = generate_sample_data()
    print(f"生成的示例数据: {data}")

    # 处理数据
    processor = DataProcessor(data)
    results = processor.process()

    # 分析结果
    set_trace()
    average = sum(results) / len(results)
    maximum = max(results)
    minimum = min(results)

    print(f"处理完成!")
    print(f"结果: {results}")
    print(f"统计信息: 平均值={average}, 最大值={maximum}, 最小值={minimum}")

if __name__ == "__main__":
    x = 10
    y = 20
    print(f"{x} + {y} =", add(x, y))
    main()
