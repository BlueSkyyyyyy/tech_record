#!/usr/bin/env bash
# 复用 kernel-opt 的 kernel_lab 容器（含 nvcc/ncu/nsys/PyTorch/flash_attn/TE）。
exec "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../../../kernel-opt/scripts/lab.sh" "$@"
