#!/usr/bin/env bash
# P68-1 实验 driver：在 kernel_lab(H100/sm90) 上跑 DeepGEMM SF layout 验收测试规范
#   （tests/test_layout.py）。直接 import 真实测试模块复用其纯 torch oracle，
#   调用真实测试体，并对拍真实 kernel / 枚举计数 / profiler 区分发射。
#   bash harness/exp_sf_layout_test_oracle.sh > harness/exp_sf_layout_test_oracle.out.txt
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
"$ROOT/scripts/lab.sh" exec "cd '$ROOT/harness' && \
  nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader | head -1 && \
  python exp_sf_layout_test_oracle.py"
