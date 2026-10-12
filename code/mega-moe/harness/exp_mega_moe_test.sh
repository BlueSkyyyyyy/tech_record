#!/usr/bin/env bash
# P41-1 实验 driver：Mega MoE 的端到端测试与逐位正确性判据（tests/test_mega_moe.py）。
#
# 为什么 H100(sm90) 可跑：
#   * 测试侧的 SF 布局变换（_cast_fp8_for_mega_moe / _to_shared_mega_moe_sf_layout /
#     _copy_fp8_sf）与逐位判据、单 rank 参考数学、开销账目都是纯 torch，架构无关；
#   * 逐位判据用真实 SM90 grouped BF16 kernel（m_grouped_bf16_gemm_nt_contiguous）演示。
# 真正端到端 fused kernel 仍是 SM100 专属，不涉及。
#   bash harness/exp_mega_moe_test.sh > harness/exp_mega_moe_test.out.txt
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="$ROOT/harness/exp_mega_moe_test.py"
"$ROOT/scripts/lab.sh" exec "nvidia-smi --query-gpu=name,compute_cap,driver_version --format=csv,noheader | head -1 \
  && python '$PY'"
