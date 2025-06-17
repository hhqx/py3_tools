

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
