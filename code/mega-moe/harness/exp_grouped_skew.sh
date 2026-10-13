#!/usr/bin/env bash
# P142-1 实验 driver：专家负载不均（token 分布偏斜）对 MoE grouped GEMM 计算侧的代价。
#   bash harness/exp_grouped_skew.sh > harness/exp_grouped_skew.out.txt
# 对应源码 exp_grouped_skew.py；参考 ~/github/DeepGEMM
#   csrc/apis/gemm.hpp:181-318（m_grouped contiguous / masked 入口）、
#   deep_gemm/include/deep_gemm/scheduler/gemm.cuh:151-156/188-225/296-309（M 块领取与跳过 gap）、
#   csrc/jit_kernels/impls/sm90_fp8_gemm_1d2d.hpp:176-283（SM90 1d2d kernel 发射）。
# 本机是 8 卡共享节点：自动绑到当前利用率最低的 GPU，避免争用噪声。
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
"$ROOT/scripts/lab.sh" exec "nvidia-smi >/dev/null 2>&1 || true; \
  IDLE=\$(nvidia-smi --query-gpu=index,utilization.gpu --format=csv,noheader,nounits | sort -t, -k2 -n | head -1 | cut -d, -f1 | tr -d ' '); \
  echo \"[exp_grouped_skew] pinned to GPU \$IDLE\"; \
  cd '$ROOT/harness' && nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader | head -1 && CUDA_VISIBLE_DEVICES=\$IDLE python exp_grouped_skew.py"
