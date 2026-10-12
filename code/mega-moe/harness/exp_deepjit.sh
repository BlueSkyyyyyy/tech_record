#!/usr/bin/env bash
# P23-1 实验 driver：在 kernel_lab(H100) 上解剖 DeepGEMM 的 DeepJIT 运行时
# （编译 / 磁盘缓存 / 加载 / 发射管线），并纯 Python 复刻其 cache key。
#   bash harness/exp_deepjit.sh > harness/exp_deepjit.out.txt
# 对应源码 exp_deepjit.py；参考 ~/github/DeepGEMM 的 csrc/runtime/jit.hpp 与
# third-party/deep_jit/include/deep_jit/{runtime,cache,backend/cuda,utils}/。
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
"$ROOT/scripts/lab.sh" exec "cd '$ROOT/harness' && nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader | head -1 && python exp_deepjit.py"
