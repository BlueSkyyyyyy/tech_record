#!/usr/bin/env bash
# P178-1 实验 driver：MoE top-k 对前向/反向成本与 grouped GEMM 效率的扫描（H100 sm90）。
#   bash harness/exp_topk_cost.sh > harness/exp_topk_cost.out.txt
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
"$ROOT/scripts/lab.sh" exec "cd '$ROOT/harness' && \
nvidia-smi --query-gpu=name,compute_cap,driver_version --format=csv,noheader | head -1 && \
python3 -u exp_topk_cost.py"
