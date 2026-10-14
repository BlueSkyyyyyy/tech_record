#!/usr/bin/env bash
# P126-1 实验 driver：SM100 sparse MQA logits 主 kernel 的架构无关逻辑验证。
#
# 为什么 H100(sm90) 可跑：主 kernel（`impls/sm100_sparse_mqa_logits.cuh`）的 UMMA/tcgen05/UTCCP/TMA 是
# SM100 专属，但本实验只抽取它**架构无关的六段**（KV smem chunk swizzle / metadata 打包解码 /
# KV 访问几何 / math warp 压缩列映射 / 资源账 + SharedStorage / SF 4x32 转置），纯整数 + 纯 bf16，
# 直接在 sm90 上编译运行真实设备代码（直接 include 真实 layout 头以核 sizeof/offsetof）。
#   bash harness/exp_sparse_mqa_logits_kernel.sh > harness/exp_sparse_mqa_logits_kernel.out.txt
# 对应源码 src/sparse_mqa_logits_kernel.cu；参考 ~/github/DeepGEMM 的
# csrc/jit_kernels/impls/sm100_sparse_mqa_logits.hpp、csrc/apis/attention.hpp、layout/sparse_mqa_logits.cuh、
# deep_gemm/include/deep_gemm/impls/sm100_sparse_mqa_logits.cuh。
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC="$ROOT/src/sparse_mqa_logits_kernel.cu"
DGEMM=${HOME}/github/DeepGEMM
"$ROOT/scripts/lab.sh" exec "nvcc -arch=sm_90a -O3 -std=c++17 --expt-relaxed-constexpr -diag-suppress 20012 \
  -I$DGEMM/deep_gemm/include -I$DGEMM/third-party/cutlass/include \
  '$SRC' -o /tmp/sparse_mqa_logits_kernel \
  && nvidia-smi --query-gpu=name,compute_cap,driver_version --format=csv,noheader | head -1 \
  && /tmp/sparse_mqa_logits_kernel"
