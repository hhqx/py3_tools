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

build: clean
	python -m build

publish:
	python -m twine upload dist/*

push:
	git push origin master:master

reinstall:
	pip install --upgrade --force-reinstall dist/*.whl

clean:
	rm -rf build/ dist/ *.egg-info .pytest_cache .mypy_cache .coverage htmlcov

# Package installation and management
.PHONY: install-dev install-full install-from-pypi uninstall

install-dev:
	pip install -e .[dev]

install-full:
	pip install -e .[full]

uninstall:
	pip uninstall -y py3_tools_hqx

install-from-pypi: uninstall
	pip install py3_tools_hqx

# Quick development cycle
.PHONY: dev
dev: uninstall clean format lint typecheck test build install-from-pypi
	@echo "Development cycle completed successfully!"