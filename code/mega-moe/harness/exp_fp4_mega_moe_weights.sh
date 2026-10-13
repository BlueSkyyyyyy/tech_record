#!/usr/bin/env bash
# P95-1 实验 driver：在 kernel_lab(H100) 上运行 exp_fp4_mega_moe_weights.py
#   —— Mega MoE 的 FP4 路由权重路径（fp8xfp4 / kPackedFP4）host+张量语义。
#   ./harness/exp_fp4_mega_moe_weights.sh
# 输出见 harness/exp_fp4_mega_moe_weights.out.txt。
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="$ROOT/harness/exp_fp4_mega_moe_weights.py"
"$ROOT/scripts/lab.sh" exec "nvidia-smi --query-gpu=name,compute_cap,driver_version --format=csv,noheader | head -1 && echo '--- PYTHON ---' && python3 '$PY'"
