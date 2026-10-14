#!/usr/bin/env bash
# P179-1 实验 driver：Mega MoE BLOCK_M 的 tile 收益（H100 sm90 代理 GEMM）。
#   bash harness/exp_block_m_tile.sh > harness/exp_block_m_tile.out.txt
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
"$ROOT/scripts/lab.sh" exec "cd '$ROOT/harness' && \
nvidia-smi --query-gpu=name,compute_cap,driver_version --format=csv,noheader | head -1 && \
python3 -u exp_block_m_tile.py"
