# 自动化脚本：一键格式化、检查、测试、打包、发布

.PHONY: all format lint typecheck test build publish clean

all: format lint typecheck test

format:
	black src/ tests/

lint:
	flake8 src/ tests/

typecheck:
	mypy src/

test:
	pytest

build:
	python -m build

publish:
	twine upload dist/*

clean:
	rm -rf build/ dist/ *.egg-info .pytest_cache .mypy_cache .coverage htmlcov
