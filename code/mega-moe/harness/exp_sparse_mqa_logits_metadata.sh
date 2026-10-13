#!/usr/bin/env bash
# P127-1 实验 driver：稀疏 MQA logits 的 metadata kernel 主循环。
#
# 为什么 H100(sm90) 可跑：`scheduler/sm100_sparse_mqa_logits_metadata.cuh` 整个 kernel 架构无关
# （只用 ptx 整数/原子、cp.async(sm80)、math::cta_exclusive_sum、cute 下标工具；UMMA/tcgen05/UTCCP/TMA
# 都在主 kernel 里）。本实验直接 include 真实头文件、用真实 launch 参数跑真实设备代码，并把输出的
# metadata 与一份独立的 host oracle（归并去重/槽打包/物理块映射/split 头/调度覆盖）逐字段对拍。
#   bash harness/exp_sparse_mqa_logits_metadata.sh > harness/exp_sparse_mqa_logits_metadata.out.txt
# 对应源码 src/sparse_mqa_logits_metadata.cu；参考 ~/github/DeepGEMM 的
# deep_gemm/include/deep_gemm/scheduler/sm100_sparse_mqa_logits_metadata.cuh、
# deep_gemm/include/deep_gemm/layout/sparse_mqa_logits.cuh、
# csrc/jit_kernels/impls/sm100_sparse_mqa_logits.hpp、csrc/apis/attention.hpp。
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC="$ROOT/src/sparse_mqa_logits_metadata.cu"
DGEMM=~/github/DeepGEMM
"$ROOT/scripts/lab.sh" exec "nvcc -arch=sm_90a -O3 -std=c++17 --expt-relaxed-constexpr -diag-suppress 20012 \
  -I$DGEMM/deep_gemm/include -I$DGEMM/third-party/cutlass/include \
  '$SRC' -o /tmp/sparse_mqa_logits_metadata \
  && nvidia-smi --query-gpu=name,compute_cap,driver_version --format=csv,noheader | head -1 \
  && /tmp/sparse_mqa_logits_metadata"
