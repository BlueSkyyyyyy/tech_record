#!/usr/bin/env bash
# P49-1 实验 driver：DeepEP 设备端公共层（common/ptx.cuh / common/math.cuh / common/utils.cuh /
#           layout/common/barrier.cuh）。
#
# 为什么 H100(sm90a) 可跑：
#   * ptx.cuh 绝大多数是架构无关内联 asm（mbarrier / TMA 1D / cp.async / 谓词化 ld-st /
#     内存序 / warp 内建）；带 `__CUDA_ARCH__>=1000` 守卫的原语在 sm90 走文件自带回退分支，
#     本实验直接验证回退分支；multimem / cvt.rs 只做 host 语义 oracle。
#   * 本文件**直接 include 真实头文件、编译运行真实设备代码**。
#   bash harness/exp_deepep_ptx_primitives.sh > harness/exp_deepep_ptx_primitives.out.txt
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC="$ROOT/src/deepep_ptx_primitives.cu"
DEP=~/github/DeepEP
"$ROOT/scripts/lab.sh" exec "nvcc -gencode arch=compute_90a,code=sm_90a -O3 -std=c++17 --expt-relaxed-constexpr -w \
  -I$DEP/deep_ep/include -I/usr/local/cuda/include/cccl \
  '$SRC' -o /tmp/deepep_ptx_primitives \
  && nvidia-smi --query-gpu=name,compute_cap,driver_version --format=csv,noheader | head -1 \
  && /tmp/deepep_ptx_primitives"
