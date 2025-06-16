# py_tools

个人常用 Python 工具集，包含调试、数据处理等实用模块。

## 目录结构

- `src/py_tools/`：主代码
- `examples/`：用例
- `tests/`：测试

## 安装

```bash
pip install .
```

## 开发环境

- Python >= 3.7
- 详见 `pyproject.toml` 的可选依赖

## 常用命令

- 格式化：`black src/ tests/`
- 静态检查：`flake8 src/ tests/`
- 类型检查：`mypy src/`
- 测试：`pytest`

## 发布

1. 构建：`python -m build`
2. 发布：`twine upload dist/*`

## License

MIT
