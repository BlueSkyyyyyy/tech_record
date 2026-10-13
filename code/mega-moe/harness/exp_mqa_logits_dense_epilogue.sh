#!/usr/bin/env bash
# P103-1 实验 driver：SM100 dense MQA logits 主 kernel 的架构无关逻辑验证。
#
# 为什么 H100(sm90) 可跑：主 kernel（`impls/sm100_mqa_logits.cuh`）的 UMMA/tcgen05/UTCCP/TMA 是
# SM100 专属，但本实验只抽取它**架构无关的五段**（dispatch_num_block_tokens / visit_tmem_head_chunks /
# reduce_pairs + store_token / issue_paged_split 均分算术 / utccp 4x32 转置），纯整数 + 纯 bf16，
# 直接在 sm90 上编译运行真实设备代码（只把 SM100 的 tmem_load 换成记录桩）。
#   bash harness/exp_mqa_logits_dense_epilogue.sh > harness/exp_mqa_logits_dense_epilogue.out.txt
# 对应源码 src/mqa_logits_dense_epilogue.cu；参考 ~/github/DeepGEMM 的
# csrc/jit_kernels/impls/sm100_mqa_logits.hpp、csrc/apis/attention.hpp、layout/mqa_logits.cuh、
# deep_gemm/include/deep_gemm/impls/sm100_mqa_logits.cuh。
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC="$ROOT/src/mqa_logits_dense_epilogue.cu"
DGEMM=~/github/DeepGEMM
"$ROOT/scripts/lab.sh" exec "nvcc -arch=sm_90a -O3 -std=c++17 --expt-relaxed-constexpr -diag-suppress 20012 \
  -I$DGEMM/deep_gemm/include -I$DGEMM/third-party/cutlass/include \
  '$SRC' -o /tmp/mqa_logits_dense_epilogue \
  && nvidia-smi --query-gpu=name,compute_cap,driver_version --format=csv,noheader | head -1 \
  && /tmp/mqa_logits_dense_epilogue"
