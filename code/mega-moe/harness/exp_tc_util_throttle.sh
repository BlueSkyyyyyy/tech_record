#!/usr/bin/env bash
# P26-1 实验 driver：在 kernel_lab(H100) 编译运行 tensor-core 利用率节流机制复刻。
#   bash harness/exp_tc_util_throttle.sh > harness/exp_tc_util_throttle.out.txt
# 对应源码 src/tc_util_throttle.cu（逻辑取自 sm100_bf16_gemm.cuh:326-343）。
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC="$ROOT/src/tc_util_throttle.cu"
"$ROOT/scripts/lab.sh" exec "nvcc -arch=sm_90 -O3 -std=c++17 '$SRC' -o /tmp/tc_util_throttle && nvidia-smi --query-gpu=name,compute_cap,driver_version --format=csv,noheader | head -1 && /tmp/tc_util_throttle"
