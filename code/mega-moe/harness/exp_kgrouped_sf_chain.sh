#!/usr/bin/env bash
# P177-1 实验 driver：把 K-grouped wgrad 的「布局 -> SF -> kernel」链路在 H100(sm90) 上端到端兑现。
#   bash harness/exp_kgrouped_sf_chain.sh > harness/exp_kgrouped_sf_chain.out.txt
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
"$ROOT/scripts/lab.sh" exec "cd '$ROOT/harness' && \
nvidia-smi --query-gpu=name,compute_cap,driver_version --format=csv,noheader | head -1 && \
python3 -u exp_kgrouped_sf_chain.py"
