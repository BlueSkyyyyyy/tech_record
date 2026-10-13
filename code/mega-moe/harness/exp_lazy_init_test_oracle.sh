#!/usr/bin/env bash
# P70-1 实验 driver：在 kernel_lab(H100/sm90) 上跑 DeepGEMM 顶层运行时的验收测试规范
#   `tests/test_lazy_init.py`（lazy init / fork safety / 原生退出清理）。
#   原样跑真实测试模块（mp 8 + unittest）、独立复刻 6 用例 LD_PRELOAD 探针矩阵、
#   再对 shutdown 幂等/终态与 env「整数」语义做利齿负样本控制。
#   bash harness/exp_lazy_init_test_oracle.sh > harness/exp_lazy_init_test_oracle.out.txt
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
"$ROOT/scripts/lab.sh" exec "cd '$ROOT/harness' && \
  nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader | head -1 && \
  python exp_lazy_init_test_oracle.py"
