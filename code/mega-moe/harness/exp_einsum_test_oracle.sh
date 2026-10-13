#!/usr/bin/env bash
# P66-1 实验 driver：在 kernel_lab(H100/sm90) 上跑 DeepGEMM einsum 家族验收测试规范。
#   直接 import 真实 tests/test_einsum.py，调用真实测试体（只打桩 bench_kineto），
#   并做 einsum 特有的利齿/盲区负样本控制与确定性开关 kernel 抓取。
#   bash harness/exp_einsum_test_oracle.sh > harness/exp_einsum_test_oracle.out.txt
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
"$ROOT/scripts/lab.sh" exec "cd '$ROOT/harness' && \
  nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader | head -1 && \
  python exp_einsum_test_oracle.py"
