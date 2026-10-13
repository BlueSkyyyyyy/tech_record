#!/usr/bin/env bash
# P98-1 实验 driver：DeepGEMM 的通用 GEMM 任务调度器 deep_gemm/include/deep_gemm/scheduler/gemm.cuh。
#
# 该文件被所有 GEMM / grouped GEMM / MoE 反向 wgrad-dgrad 复用，含 `Scheduler` 结构体与
# `get_num_1d_blocks_per_group`。它的设备方法全是整数索引运算（不碰 UMMA/TMA），故可在 H100(sm90)
# 上直接 include 真实头文件、编译运行真实设备代码。唯一的架构分支 `#if __CUDA_ARCH__ < 1000`
# 是 SM90 的「不齐整 TMA multicast」修补；sm90 运行即走该分支，与 H100 上真实反向 grouped GEMM 一致。
#   A) get_num_1d_blocks_per_group 枚举启发式 vs 暴力
#   B) get_swizzled_block_idx 的 L2 swizzle + SM90 multicast 修补：穷举 vs 独立复刻
#   C) 六种 GemmType 的 get_next_block 覆盖性（[0,N) 排列）+ k-grouped 物理 K 起点/SF-K 前缀和
#   D) get_aligned_effective_m_in_block 的 psum 尾块 16 对齐
#   E) is_computation_valid（masked/contiguous/psum）
#   F) is_tma_multicast_valid（SM90 multicast 对端一致性）
#   G) get_global_idx 对 7 种 GemmType × {MN,K,SF_K} 的偏移
#   bash harness/exp_gemm_scheduler.sh > harness/exp_gemm_scheduler.out.txt
# 对应源码 src/gemm_scheduler.cu。同一 .cu 同时编 sm_90a 与 sm_100a（后者仅编译期覆盖 SM100 分支）。
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC="$ROOT/src/gemm_scheduler.cu"
DGEMM=~/github/DeepGEMM
"$ROOT/scripts/lab.sh" exec "set -e
  echo '-- 环境坐标 --'
  ( cd $DGEMM && git -c safe.directory='*' log --oneline -1 )
  nvidia-smi --query-gpu=name,compute_cap,driver_version --format=csv,noheader | head -1
  nvcc --version | tail -1
  sha256sum $DGEMM/deep_gemm/include/deep_gemm/scheduler/gemm.cuh
  wc -l $DGEMM/deep_gemm/include/deep_gemm/scheduler/gemm.cuh
  echo
  echo '-- 编译（sm_90a 运行 + sm_100a 仅编译覆盖 SM100 分支）--'
  nvcc -std=c++20 -O2 -w --expt-relaxed-constexpr \
    -gencode arch=compute_90a,code=sm_90a \
    -gencode arch=compute_100a,code=sm_100a \
    -I$DGEMM/deep_gemm/include -I$DGEMM/third-party/cutlass/include \
    '$SRC' -o /tmp/gemm_scheduler
  echo 'build OK (sm90a + sm100a)'
  echo
  /tmp/gemm_scheduler"
