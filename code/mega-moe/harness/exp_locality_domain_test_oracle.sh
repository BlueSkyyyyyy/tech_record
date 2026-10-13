#!/usr/bin/env bash
# P71-1 实验 driver：在 kernel_lab(H100/sm90) 上跑 SM 局部域的验收测试规范
#   `tests/test_locality_domain.py`（域表 / 均衡 / MLOPart 生命周期 / 无 MLOPart 回退）。
#   原样跑两个可跑的用例、独立复刻 get_even/balance、再对契约负样本做利齿控制。
#   bash harness/exp_locality_domain_test_oracle.sh > harness/exp_locality_domain_test_oracle.out.txt
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
"$ROOT/scripts/lab.sh" exec "cd '$ROOT/harness' && \
  nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader | head -1 && \
  python exp_locality_domain_test_oracle.py"
