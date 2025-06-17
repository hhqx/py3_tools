from py3_tools.py_debug import Debugger

@Debugger.attach_on_error()
def risky_operation():
    # 任何可能抛异常的逻辑
    result = 1 / 0  # 故意制造异常

if __name__ == "__main__":
    risky_operation()