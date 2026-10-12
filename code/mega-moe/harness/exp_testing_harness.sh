#!/usr/bin/env bash
# P30-1 实验 driver：在 kernel_lab(H100) 上跑 DeepGEMM 测试基座
#   deep_gemm/testing/{bench,numeric,utils}.py + deep_gemm/utils/dist.py 的语义与实测。
#   bash harness/exp_testing_harness.sh > harness/exp_testing_harness.out.txt
# 对应源码 exp_testing_harness.py；参考 ~/github/DeepGEMM 的
# deep_gemm/testing/、deep_gemm/utils/dist.py、tests/test_mega_moe.py:354、tests/test_bf16.py:45/56。
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
"$ROOT/scripts/lab.sh" exec "cd '$ROOT/harness' && nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader | head -1 && python exp_testing_harness.py"
