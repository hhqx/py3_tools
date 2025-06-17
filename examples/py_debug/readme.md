# Python Debugging Utilities

## 核心功能

`py_debug` 是一个的 Python 调试工具，专为解决 Python 应用中的异常捕获与实时调试而设计。它可以：

- **保护异常现场**：在异常发生时，自动捕获并保存完整的调用栈和上下文
- **动态接入调试器**：无需修改代码，通过环境变量动态开启调试模式
- **分布式调试支持**：针对 PyTorch 分布式应用提供特殊支持，区分处理不同 rank 进程
- **多种调试模式**：支持控制台(console)、网页(web)和套接字(socket)三种调试接口
- **简洁易用**：通过简单的装饰器和环境变量，即可接入强大的调试功能

## 使用场景

1. **开发调试阶段**：在开发新功能时，快速定位和排查错误
2. **生产环境故障排查**：通过设置环境变量，临时开启调试模式排查问题
3. **分布式应用调试**：解决多进程分布式应用（如 PyTorch 分布式训练）中的调试难题
4. **远程服务器调试**：使用 Web 或 Socket 调试接口，远程连接到服务器进程进行调试

## Install
```shell
git clone https://github.com/hhqx/py3_tools.git
cd py3_tools
pip install -e .[py_debug]
```

## 快速上手

### 1. 装饰器使用方式

使用 `@Debugger.attach_on_error()` 装饰可能出错的函数，在异常发生时自动进入调试模式：

```python
from py3_tools.py_debug import Debugger

@Debugger.attach_on_error()
def my_function():
    x = 10
    y = 0
    return x / y  # 这里会触发 ZeroDivisionError
```

### 2. 通过环境变量启用调试

无需修改代码，通过环境变量动态控制是否开启调试：

```bash
# 开启调试模式
export IPDB_DEBUG=1

# 选择调试模式：console(默认), web, socket
export IPDB_MODE=console

# 运行程序
python your_script.py
```

### 3. 分布式应用调试

在 PyTorch 分布式环境中，不同 rank 使用不同的调试方式：

```python
import torch.distributed as dist
from py3_tools.py_debug import Debugger

dist.init_process_group(backend='nccl')
rank = dist.get_rank()

@Debugger.attach_on_error()
def process_data():
    if rank == 1:
        # rank 1 会触发错误，自动启动调试器
        x = [1, 2, 3][10]  # 索引越界错误
    return "Success"
```

### 4. 上下文异常调试

可以使用 try/except 块和 Debugger 方法来调试特定代码块：

```python
from py3_tools.py_debug import Debugger
import sys

def risky_function():
    try:
        print("执行风险操作...")
        result = 1 / 0
        return result
    except Exception as e:
        if Debugger.debug_flag:
            print(f"捕获到异常: {e}")
            _, tb = sys.exc_info()[1], sys.exc_info()[2]
            Debugger.blocking_console_post_mortem(rank=0)
        else:
            raise
```

## 详细使用说明

### 单进程调试

装饰任何可能出错的函数，当异常发生并且 `IPDB_DEBUG=1` 时，将自动在异常位置进入调试会话：

```python
from py3_tools.py_debug import Debugger

# 通过命令行参数启用调试
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--debug', action='store_true')
parser.add_argument('--debug_mode', choices=['console', 'web', 'socket'])
args = parser.parse_args()

if args.debug:
    Debugger.debug_flag = True
if args.debug_mode:
    Debugger.debug_mode = args.debug_mode

@Debugger.attach_on_error()
def complex_calculation():
    # 一些可能出错的代码
    result = process_data()
    return analyze_result(result)
```

### 分布式 PyTorch 调试

针对分布式训练，系统会自动处理不同 rank 的调试方式：

```python
@Debugger.attach_on_error()
def train_epoch(model, dataloader):
    for batch in dataloader:
        outputs = model(batch)
        loss = compute_loss(outputs)
        loss.backward()
        # 如果这里出现错误，根据 debug_mode 和 rank 不同采取不同调试方式:
        # - console 模式: rank 0 直接在控制台调试，其他 rank 暂停等待
        # - web 模式: 每个 rank 在端口 4444+rank 启动 web-pdb 服务器
        # - socket 模式: 每个 rank 创建 Unix 套接字等待调试客户端连接
```

### 调试模式说明

系统支持三种调试模式，可通过环境变量 `IPDB_MODE` 或代码中设置 `Debugger.debug_mode` 来选择：

1. **console**: 
   - rank 0 使用标准控制台调试，其他 rank 暂停等待
   - 适合单机开发调试

2. **web**:
   - 每个 rank 启动独立的 web-pdb 服务器
   - 服务器端口: `4444 + rank` 
   - 通过浏览器访问 `http://hostname:port/` 进行调试
   - 适合远程开发环境

3. **socket (默认)**:
   - 每个 rank 创建 Unix 套接字 `/tmp/pdb.sock.{rank}`
   - 调试客户端可通过 `nc -U /tmp/pdb.sock.{rank}` 或 `socat - UNIX-CONNECT:/tmp/pdb.sock.{rank}` 连接
   - 适合无 GUI 环境或需要自定义调试客户端的场景

## 实现细节
1. **异常捕获机制**：
   - 使用装饰器拦截函数执行过程中的异常。
   - 检查调试模式是否开启（环境变量或标志）。
   - 获取异常信息和调用栈，准备调试环境。

2. **调试器启动逻辑**：
   - 单进程：直接使用 `ipdb.post_mortem()` 在异常位置启动交互式调试。
   - 多进程：根据 rank 和 debug_mode 决定调试方式：
     - console 模式: rank 0 使用 ipdb，其他 rank 阻塞等待
     - web 模式: 所有 rank 启动 web-pdb 服务器在不同端口
     - socket 模式: 所有 rank 创建 Unix 套接字等待连接

3. **环境变量控制**：
   - 通过 `IPDB_DEBUG=1` 开启调试模式。
   - 通过 `IPDB_MODE=[console|web|socket]` 选择调试模式。
   - 可通过命令行参数覆盖环境变量设置。

## 示例脚本
### 单进程调试脚本
[debug_single_process.py](debug_single_process.py): 演示各种单进程调试场景。
```shell
# 基本用法
export IPDB_DEBUG=1
python examples/py_debug/debug_single_process.py --mode error

# 使用上下文管理器进行调试
python examples/py_debug/debug_single_process.py --mode context --debug

# 使用 web 模式调试
export IPDB_MODE=web
python examples/py_debug/debug_single_process.py --mode math_error --debug
```

### 分布式调试脚本
[debug_multi_torch_rank.py](debug_multi_torch_rank.py): 演示不同错误类型和调试模式的分布式调试。
```shell
# 基本用法（默认 console 模式）
export IPDB_DEBUG=1
torchrun --nnodes=1 --nproc_per_node=3 examples/py_debug/debug_multi_torch_rank.py --fail_ranks 0 2

# 使用 socket 模式调试
export IPDB_DEBUG=1
export IPDB_MODE=socket
torchrun --nnodes=1 --nproc_per_node=3 examples/py_debug/debug_multi_torch_rank.py --fail_ranks 1
# 在另一个终端中：nc -U /tmp/pdb.sock.1

# 测试不同类型的错误
torchrun --nnodes=1 --nproc_per_node=2 examples/py_debug/debug_multi_torch_rank.py --fail_ranks 0 --error_type zerodivision --debug
```

## 注意事项
- **分布式调试**：
  - `console` 模式: 只有 `rank 0` 进行交互式调试，其他 rank 暂停等待。
  - `web` 模式: 各 rank 在端口 `4444 + rank` 启动独立服务器。
  - `socket` 模式: 各 rank 创建套接字 `/tmp/pdb.sock.{rank}` 等待连接。
- **环境变量**：
  - `IPDB_DEBUG=1`: 启用调试功能。
  - `IPDB_MODE=[console|web|socket]`: 设置调试模式。
  - 使用 `torchrun` 时需设置正确的分布式环境。
- **调试客户端**：
  - Socket 模式推荐使用 `nc -U /tmp/pdb.sock.{rank}` 或 `socat - UNIX-CONNECT:/tmp/pdb.sock.{rank}` 连接。


## 工作原理

`py_debug` 工具采用装饰器模式捕获异常，并根据环境提供合适的调试接口。下面通过流程图和时序图来解释其工作原理。

### 1. 异常捕获流程

装饰器 `@Debugger.attach_on_error()` 包装函数，当异常发生时，根据调试标志和运行环境决定如何处理：

```mermaid
flowchart TD
    A[用户代码调用被装饰函数] --> B{执行函数}
    B -->|正常执行| C[返回结果]
    B -->|发生异常| D{是否启用调试?}
    D -->|否| E[正常抛出异常]
    D -->|是| F{是否为分布式环境?}
    F -->|否| G[启动本地 ipdb 调试会话]
    F -->|是| H{是否为 rank 0?}
    H -->|是| G[启动本地 ipdb 调试会话]
    H -->|否| I{选择的调试模式?}
    I -->|console| J[阻塞等待 rank 0 完成调试]
    I -->|web| K[启动 web-pdb 服务器 端口=4444+rank]
    I -->|socket| L[创建 Unix 套接字等待连接]
    K --> M[在浏览器访问调试界面]
    L --> N[使用 nc/socat 连接套接字]
    G --> O[完成调试]
    J --> O
    M --> O
    N --> O
    O --> E
```

### 2. 调试器启动过程

当异常发生并且调试模式已启用时，系统按照以下步骤启动调试器：

```mermaid
sequenceDiagram
    participant 用户代码
    participant Debugger
    participant IPDB
    participant WebPdb
    participant Socket
    
    用户代码->>Debugger: 调用被装饰函数
    Debugger->>用户代码: 执行函数内容
    
    alt 发生异常
        rect rgb(255, 240, 240)
        用户代码->>Debugger: 抛出异常
        Debugger->>Debugger: 检查调试模式是否启用
        
        alt 调试模式已启用
            rect rgb(240, 255, 240)
            Debugger->>Debugger: 检查是否为分布式环境
            
            alt 单进程环境或 rank == 0
                rect rgb(240, 240, 255)
                Debugger->>IPDB: 调用 post_mortem()
                IPDB->>用户代码: 在异常位置启动调试会话
                end
            else rank > 0
                rect rgb(255, 245, 225)
                Debugger->>Debugger: 检查调试模式
                
                alt debug_mode == 'console'
                    rect rgb(230, 245, 255)
                    Debugger->>用户代码: 阻塞等待 rank 0 完成调试
                    end
                else debug_mode == 'web'
                    rect rgb(245, 230, 255)
                    Debugger->>WebPdb: 启动服务器(端口=4444+rank)
                    WebPdb->>用户代码: 提供 Web 接口进行调试
                    end
                else debug_mode == 'socket'
                    rect rgb(255, 235, 230)
                    Debugger->>Socket: 创建 Unix 套接字 /tmp/pdb.sock.{rank}
                    Socket->>用户代码: 等待调试客户端连接
                    end
                end
                end
            end
            end
        else 调试模式未启用
            rect rgb(245, 245, 245)
            Debugger->>用户代码: 继续抛出异常
            end
        end
        end
    else 正常执行
        rect rgb(230, 255, 230)
        用户代码->>Debugger: 函数执行完成
        Debugger->>用户代码: 返回结果
        end
    end
```

### 3. 实现机制

- **装饰器模式**: `@Debugger.attach_on_error()` 拦截函数执行并捕获异常
- **环境检测**: 通过环境变量 `IPDB_DEBUG` 或命令行参数 `--debug` 启用调试
- **分布式感知**: 检测 PyTorch 分布式环境并获取当前进程的 rank
- **异常现场保护**: 保留完整调用栈和变量信息，不丢失异常上下文
- **多种调试界面**: 
  - console 模式: rank 0 使用标准 ipdb 交互调试，其他 rank 等待
  - web 模式: 使用 web-pdb 提供 Web 界面，便于远程调试
  - socket 模式: 创建 Unix 套接字，允许任意客户端连接，最大灵活性
