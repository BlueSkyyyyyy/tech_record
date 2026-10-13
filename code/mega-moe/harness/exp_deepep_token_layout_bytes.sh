#!/usr/bin/env bash
# P152-1 实验 driver：DeepEP 真实 per-token 缓冲布局字节口径。
#
# 为什么 H100(sm90) 可跑：
#   * 直接 include 真实 `deep_ep/layout/ep/token.cuh`（全 host/device 纯算术，无 SM100 指令、
#     无 NCCL/运行时句柄）；dispatch/combine 布局函数体按 csrc/kernels/ep/{dispatch,combine}.hpp
#     原样重述（那两个头 include 了 <nccl.h>，故只重述两行调用）。
#   bash harness/exp_deepep_token_layout_bytes.sh > harness/exp_deepep_token_layout_bytes.out.txt
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC="$ROOT/src/deepep_token_layout_bytes.cu"
DEP=~/github/DeepEP
"$ROOT/scripts/lab.sh" exec "nvcc -gencode arch=compute_90a,code=sm_90a -O3 -std=c++17 --expt-relaxed-constexpr -w \
  -I$DEP/deep_ep/include -I/usr/local/cuda/include/cccl \
  '$SRC' -o /tmp/deepep_token_layout_bytes \
  && nvidia-smi --query-gpu=name,compute_cap,driver_version --format=csv,noheader | head -1 \
  && /tmp/deepep_token_layout_bytes"
