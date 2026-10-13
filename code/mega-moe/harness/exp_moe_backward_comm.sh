#!/usr/bin/env bash
# P94-1 实验 driver：在 kernel_lab(H100, 8 卡) 上运行 harness/exp_moe_backward_comm.py，
# 用真实 ragged all-to-all 量 MoE 反向的两个通信原语（dispatch/combine 的伴随）。
#   ./harness/exp_moe_backward_comm.sh
# 输出见 harness/exp_moe_backward_comm.out.txt。
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="$ROOT/harness/exp_moe_backward_comm.py"
"$ROOT/scripts/lab.sh" exec "nvidia-smi --query-gpu=name,compute_cap,driver_version --format=csv,noheader | head -1 && echo '--- PYTHON ---' && NCCL_NVLS_ENABLE=0 python3 '$PY'"
