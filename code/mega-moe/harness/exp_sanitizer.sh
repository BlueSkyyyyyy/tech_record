#!/usr/bin/env bash
# P60-1 实验 driver：在 kernel_lab(H100/sm90) 上跑 DeepGEMM 的内存安全验证层
#   tests/test_sanitizer.py（compute-sanitizer 驱动）。
#   bash harness/exp_sanitizer.sh > harness/exp_sanitizer.out.txt
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
"$ROOT/scripts/lab.sh" exec "cd '$ROOT/harness' && nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader | head -1 && \
  /usr/local/cuda/bin/compute-sanitizer --version 2>&1 | head -2 && python exp_sanitizer.py"
