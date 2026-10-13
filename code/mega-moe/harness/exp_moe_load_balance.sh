#!/usr/bin/env bash
# P129-1 实验 driver：在 kernel_lab(H100, sm90) 上运行 MoE 路由负载均衡三件套
# （aux-loss / router z-loss / aux-loss-free bias）的数值与梯度实验。
#   ./harness/exp_moe_load_balance.sh
# 输出见 harness/exp_moe_load_balance.out.txt。
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="$ROOT/harness/exp_moe_load_balance.py"
"$ROOT/scripts/lab.sh" exec "nvidia-smi --query-gpu=name,compute_cap,driver_version --format=csv,noheader | head -1 && echo '--- PYTHON ---' && python3 '$PY'"
