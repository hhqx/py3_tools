#!/usr/bin/env python3
import sys
import pdb

class CustomPdb(pdb.Pdb):
    def __init__(self, completekey=None, stdin=None, stdout=None, **kwargs):
        # 调用父类构造函数，确保所有参数都被正确处理
        super().__init__(completekey=completekey, stdin=stdin, stdout=stdout, **kwargs)
        # 可选：修改提示符
        self.prompt = '(custom-pdb) '

def main():
    # 创建调试器实例，指定 Tab 补全、标准输入输出
    debugger = CustomPdb(
        completekey='tab',
        stdin=sys.stdin,
        stdout=sys.stdout
    )

    # 演示代码
    x = 10
    y = 20
    z = x + y

    # 在这里设断点，进入自定义调试器
    debugger.set_trace()

    print(f"{x} + {y} = {z}")

if __name__ == '__main__':
    main()
