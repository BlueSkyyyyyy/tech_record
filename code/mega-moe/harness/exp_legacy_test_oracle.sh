#!/usr/bin/env bash
# P72-1 实验 driver：在 kernel_lab(H100/sm90) 上跑 DeepGEMM legacy Triton grouped GEMM
#   验收测试规范 tests/test_legacy.py。直接 import 真实测试模块并调用真实测试体
#   （只打桩 bench_kineto），另做闸门/别名/expand/判据利齿的独立复核。
#   bash harness/exp_legacy_test_oracle.sh > harness/exp_legacy_test_oracle.out.txt
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
"$ROOT/scripts/lab.sh" exec "cd '$ROOT/harness' && \
  nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader | head -1 && \
  python exp_legacy_test_oracle.py"
