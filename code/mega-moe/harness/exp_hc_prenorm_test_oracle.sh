#!/usr/bin/env bash
# P69-1 实验 driver：在 kernel_lab(H100/sm90) 上跑 HyperConnection prenorm GEMM 验收测试规范。
#   直接 import 真实 tests/test_hyperconnection.py，调用真实测试体（只打桩 bench_kineto），
#   再用真实 bench_kineto 抓测试自带的 24 行吞吐表，并做 TF32/契约/split 的利齿负样本控制。
#   bash harness/exp_hc_prenorm_test_oracle.sh > harness/exp_hc_prenorm_test_oracle.out.txt
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
"$ROOT/scripts/lab.sh" exec "cd '$ROOT/harness' && \
  nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader | head -1 && \
  python exp_hc_prenorm_test_oracle.py"
