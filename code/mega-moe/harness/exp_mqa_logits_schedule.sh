#!/usr/bin/env bash
# P13-1 实验 driver：SM100 dense MQA logits 的连续版 metadata 负载均衡调度器。
#
# 为什么 H100(sm90) 可跑：`deep_gemm/include/deep_gemm/scheduler/sm100_mqa_logits.cuh`
# 的 `sm100_mqa_logits_metadata` kernel 与 `SM100MQALogitsScheduler` 不依赖 UMMA/tcgen05，
# 直接编译运行真实设备代码（主 kernel `impls/sm100_mqa_logits.cuh` 仍是 SM100 专属，不涉及）。
#   (A) 真实 metadata kernel 生成 schedule_meta；(B) host 独立复刻对拍 spans/边界；
#   (C) 真实 scheduler 发射任务，验证「所有有效 split 恰好覆盖一次」；
#   (D) 量化 scheduled vs grid-stride 的 per-SM 负载均衡。
#   bash harness/exp_mqa_logits_schedule.sh > harness/exp_mqa_logits_schedule.out.txt
# 对应源码 src/mqa_logits_schedule.cu；参考 ~/github/DeepGEMM 的
# csrc/apis/attention.hpp:46-134、csrc/jit_kernels/impls/sm100_mqa_logits.hpp:11-115,
# deep_gemm/include/deep_gemm/scheduler/sm100_mqa_logits.cuh、
# deep_gemm/include/deep_gemm/impls/sm100_mqa_logits.cuh、layout/mqa_logits.cuh。
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC="$ROOT/src/mqa_logits_schedule.cu"
DGEMM=${HOME}/github/DeepGEMM
"$ROOT/scripts/lab.sh" exec "nvcc -arch=sm_90a -O3 -std=c++17 --expt-relaxed-constexpr -diag-suppress 20012 \
  -I$DGEMM/deep_gemm/include -I$DGEMM/third-party/cutlass/include \
  '$SRC' -o /tmp/mqa_logits_schedule \
  && nvidia-smi --query-gpu=name,compute_cap,driver_version --format=csv,noheader | head -1 \
  && /tmp/mqa_logits_schedule"
