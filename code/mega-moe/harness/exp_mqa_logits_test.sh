#!/usr/bin/env bash
# P50-1 实验 driver：DeepGEMM MQA logits 的端到端验收测试与正确性 oracle
# (`tests/test_attention.py`)。
#
# 为什么 H100(sm90) 能真跑：
#   * MQA logits 的 SM90 实现（fmt='fp8'，fp32 logits）真实存在于
#     `impls/sm90_fp8_mqa_logits.cuh` / `sm90_fp8_paged_mqa_logits.cuh`；
#   * 本脚本直接 import 真实 tests/test_attention.py，调用真实 `test_mqa_logits()` /
#     `test_paged_mqa_logits()`（SM100 的 mxfp4/mxfp8 与 sparse 用例由 @test_filter 跳过），
#     再独立验证 ref_mqa_logits / to_mqa_weights / quantize_mqa_q / KV 融合布局 / 判据工具。
#   * 只限制用例数（DG_MQA_NUM_CASES=2）让输出有界。
#   bash harness/exp_mqa_logits_test.sh > harness/exp_mqa_logits_test.out.txt
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="$ROOT/harness/exp_mqa_logits_test.py"
"$ROOT/scripts/lab.sh" exec "nvidia-smi --query-gpu=name,compute_cap,driver_version --format=csv,noheader | head -1 \
  && python '$PY'"
