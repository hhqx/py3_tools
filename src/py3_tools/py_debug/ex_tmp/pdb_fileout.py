#!/usr/bin/env python3
import sys
import pdb

class CustomPdb(pdb.Pdb):
    def __init__(self, completekey=None, stdin=None, stdout=None, **kwargs):
        super().__init__(completekey=completekey, stdin=stdin, stdout=stdout, **kwargs)
        # 修改提示符
        self.prompt = '(custom-pdb) '

def main():
    # 打开日志文件，调试输出会写入这里
    log_file = open('pdb_output.log', 'w+')
    try:
        debugger = CustomPdb(
            completekey='tab',
            stdin=sys.stdin,
            stdout=log_file
        )

        # 演示代码
        x = 10
        y = 20
        z = x + y

        # 进入调试器（输出会写入 pdb_output.log）
        debugger.set_trace()

        print(f"{x} + {y} = {z}")
    finally:
        log_file.close()

if __name__ == '__main__':
    main()
