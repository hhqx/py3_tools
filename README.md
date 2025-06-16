# py3_tools

个人常用 Python 工具集，包含调试、Gitee PR 统计等实用模块，适用于开发、数据分析和分布式调试等场景。

## 工具目录

| 工具模块         | 简介                           | 使用说明/示例                      |
|------------------|-------------------------------|-------------------------------------|
| py_debug         | Python 调试增强工具，支持单进程和分布式调试 | [py_debug 使用说明](https://github.com/hhqx/py3_tools/tree/master/examples/py_debug/readme.md) |
| gitee_pr_utils   | Gitee PR 查询与统计工具        | [gitee_pr_stat 使用说明](https://github.com/hhqx/py3_tools/tree/master/examples/gitee/readme.md) |

## 安装

```bash
git clone https://github.com/hhqx/py3_tools.git
cd py3_tools
pip install -e .[dev,py_debug,gitee]
```

- 只需调试工具：`pip install -e .[py_debug]`
- 只需 Gitee 工具：`pip install -e .[gitee]`

## 快速入口

- [单进程调试工具示例](https://github.com/hhqx/py3_tools/tree/master/examples/py_debug/readme.md#单进程调试)，[debug_single_process.py](https://github.com/hhqx/py3_tools/tree/master/examples/py_debug/debug_single_process.py)
- [分布式调试示例](https://github.com/hhqx/py3_tools/tree/master/examples/py_debug/readme.md#分布式调试脚本), [debug_multi_torch_rank.py](https://github.com/hhqx/py3_tools/tree/master/examples/py_debug/debug_multi_torch_rank.py)
- [Gitee PR 统计示例](https://github.com/hhqx/py3_tools/tree/master/examples/gitee/readme.md)


## License

MIT