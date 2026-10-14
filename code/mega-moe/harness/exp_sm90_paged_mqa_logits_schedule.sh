#!/usr/bin/env bash
# P20-1 实验 driver：SM90 paged MQA logits 的 metadata 负载均衡调度器。
#
# 为什么 H100(sm90) 可跑：`deep_gemm/include/deep_gemm/scheduler/sm90_paged_mqa_logits.cuh`
# 的 `sm90_paged_mqa_logits_metadata` kernel 与 `SM90PagedMQALogitsScheduler`
# 只依赖 `<deep_gemm/common/math.cuh>` / `<deep_gemm/ptx/utils.cuh>` 与 PDL，
# 不依赖 WGMMA/TMA/UMMA，因此直接编译运行【真实设备代码】。
#   (A) 真实 metadata kernel 生成 schedule_meta；
#   (B) host 独立复刻对拍每 SM 边界（varlen atom 分组 / 前缀和 / 反序分配 / 二分）；
#   (C) 真实 scheduler 发射任务，验证「所有 (atom, split) 恰好覆盖一次」+ num_kv/advance/映射。
#   bash harness/exp_sm90_paged_mqa_logits_schedule.sh > harness/exp_sm90_paged_mqa_logits_schedule.out.txt
# 对应源码 src/sm90_paged_mqa_logits_schedule.cu；参考 ~/github/DeepGEMM 的
# csrc/apis/attention.hpp:136-172、csrc/jit_kernels/impls/sm90_fp8_mqa_logits.hpp:94-135、
# deep_gemm/include/deep_gemm/scheduler/sm90_paged_mqa_logits.cuh、
# deep_gemm/include/deep_gemm/impls/sm90_fp8_paged_mqa_logits.cuh:137-140。
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC="$ROOT/src/sm90_paged_mqa_logits_schedule.cu"
DGEMM=${HOME}/github/DeepGEMM
"$ROOT/scripts/lab.sh" exec "nvcc -arch=sm_90a -O3 -std=c++17 --expt-relaxed-constexpr -diag-suppress 20012 \
  -I$DGEMM/deep_gemm/include -I$DGEMM/third-party/cutlass/include \
  '$SRC' -o /tmp/sm90_paged_mqa_logits_schedule \
  && nvidia-smi --query-gpu=name,compute_cap,driver_version --format=csv,noheader | head -1 \
  && /tmp/sm90_paged_mqa_logits_schedule"
