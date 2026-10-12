#!/usr/bin/env bash
# P9-1 实验 driver：在 kernel_lab(H100/sm90) 上跑 DeepGEMM 稀疏注意力 indexer 的 MQA logits。
#   `fp8_fp4_mqa_logits`（连续）与 `fp8_fp4_paged_mqa_logits`（分页）实测。
#   bash harness/exp_mqa_logits.sh > harness/exp_mqa_logits.out.txt
# 对应源码 exp_mqa_logits.py；参考 ~/github/DeepGEMM
#   csrc/apis/attention.hpp、csrc/jit_kernels/impls/sm90_fp8_mqa_logits.hpp、
#   deep_gemm/include/deep_gemm/impls/sm90_fp8_mqa_logits.cuh / sm90_fp8_paged_mqa_logits.cuh。
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
"$ROOT/scripts/lab.sh" exec "cd '$ROOT/harness' && nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader | head -1 && python exp_mqa_logits.py"
