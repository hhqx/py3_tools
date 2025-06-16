#!/bin/bash
# 自动化一键发布脚本
set -e

make clean
make all
make build

echo "Ready to publish. Run 'make publish' to upload to PyPI."
