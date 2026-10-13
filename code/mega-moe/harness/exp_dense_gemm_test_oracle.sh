#!/usr/bin/env bash
# P64-1 实验 driver：在 kernel_lab(H100/sm90) 上跑 DeepGEMM 稠密 GEMM 验收测试规范。
#   直接 import 真实 tests/test_fp8_fp4.py + tests/test_bf16.py，调用真实测试体
#   （只缩小形状枚举、打桩 bench_kineto），并做 calc_diff 盲区负样本控制。
#   bash harness/exp_dense_gemm_test_oracle.sh > harness/exp_dense_gemm_test_oracle.out.txt
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
"$ROOT/scripts/lab.sh" exec "cd '$ROOT/harness' && \
  nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader | head -1 && \
  python exp_dense_gemm_test_oracle.py"
