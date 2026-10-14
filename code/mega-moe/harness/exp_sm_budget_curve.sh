#!/usr/bin/env bash
# P180-1 实验 driver：MoE grouped GEMM 的「SM 预算曲线」。
#   bash harness/exp_sm_budget_curve.sh > harness/exp_sm_budget_curve.out.txt
# 对应源码 exp_sm_budget_curve.py；参考 ~/github/DeepGEMM
#   deep_gemm/include/deep_gemm/scheduler/gemm.cuh:12-27（kNumSMs 决定 L2 swizzle 分组宽度）、
#   :180（persistent grid 步长 = kNumSMs）、csrc/runtime/runtime.hpp:78-87（set_num_sms）、
#   csrc/jit_kernels/impls/sm90_fp8_gemm_1d2d.hpp:92（.num_sms = runtime->get_num_sms()）。
# 本机是 8 卡共享节点：自动绑到当前利用率最低的 GPU。
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
"$ROOT/scripts/lab.sh" exec "nvidia-smi >/dev/null 2>&1 || true; \
  IDLE=\$(nvidia-smi --query-gpu=index,utilization.gpu --format=csv,noheader,nounits | sort -t, -k2 -n | head -1 | cut -d, -f1 | tr -d ' '); \
  echo \"[exp_sm_budget_curve] pinned to GPU \$IDLE\"; \
  cd '$ROOT/harness' && nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader | head -1 && CUDA_VISIBLE_DEVICES=\$IDLE python exp_sm_budget_curve.py"
