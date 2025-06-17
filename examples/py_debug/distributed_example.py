import torch.distributed as dist
from py3_tools.py_debug import Debugger

dist.init_process_group(backend='gloo')
rank = dist.get_rank()

@Debugger.attach_on_error()
def train_step():
    print(f"Process rank {rank} running train_step()")
    if rank == 1:
        raise RuntimeError("模拟错误发生于进程 rank 1")

if __name__ == '__main__':
    train_step()