import sys
from functools import wraps

__ALL__ = ['debug_on_exception']


def debug_on_exception(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception:
            import ipdb
            _, _, tb = sys.exc_info()
            ipdb.post_mortem(tb)
            raise  # 可选：调试后继续抛出异常
    return wrapper

@debug_on_exception
def buggy_func():
    a = 1 / 0  # 发生异常时自动进入 ipdb

# 示例函数，调用时会触发异常并进入调试模式
# 你可以在这里添加更多的函数或逻辑来测试调试功能
if __name__ == "__main__":
    buggy_func()