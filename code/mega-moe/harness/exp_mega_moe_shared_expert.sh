#!/usr/bin/env bash
# P34-1 实验 driver：Mega MoE 的共享专家（shared expert）分支。
#
# 为什么 H100(sm90) 可跑：
#   * host 编排层（get_symm_buffer_size_for_mega_moe / MegaMoEBuffer）纯 host、架构无关；
#   * 调度器 scheduler/mega_moe.cuh 的任务枚举器/constexpr 公式不依赖 UMMA，可编译真实设备代码；
#   * 共享专家的数学（x@W1ᵀ→SwiGLU→@W2ᵀ，无权重，加进 combine）架构无关，用真实 SM90 BF16 kernel 跑。
# 主 kernel impls/sm100_fp8_fp4_mega_moe.cuh 仍是 SM100 专属，不涉及。
#   bash harness/exp_mega_moe_shared_expert.sh > harness/exp_mega_moe_shared_expert.out.txt
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC="$ROOT/src/mega_moe_shared_schedule.cu"
PY="$ROOT/harness/exp_mega_moe_shared_expert.py"
DGEMM=${HOME}/github/DeepGEMM
"$ROOT/scripts/lab.sh" exec "nvcc -arch=sm_90a -O3 -std=c++17 --expt-relaxed-constexpr -w \
  -diag-suppress 20012 -diag-suppress 2388 -diag-suppress 2389 \
  -I$DGEMM/deep_gemm/include -I$DGEMM/third-party/cutlass/include \
  '$SRC' -o /tmp/mega_moe_shared_schedule \
  && nvidia-smi --query-gpu=name,compute_cap,driver_version --format=csv,noheader | head -1 \
  && /tmp/mega_moe_shared_schedule \
  && echo '' && echo '=== Python: host 对拍 + 调度语义 + UTCCP 映射 + H100 数值 ===' \
  && python '$PY'"
