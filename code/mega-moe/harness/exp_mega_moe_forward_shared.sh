#!/usr/bin/env bash
# P158-1 实验 driver：在 kernel_lab(H100) 上运行 harness/exp_mega_moe_forward_shared.py，
# 用 SM90 grouped FP8（路由）+ 稠密 FP8（共享）复刻「路由 + 共享 + combine」整条 Mega MoE 前向。
#   ./harness/exp_mega_moe_forward_shared.sh
# 输出见 harness/exp_mega_moe_forward_shared.out.txt。
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="$ROOT/harness/exp_mega_moe_forward_shared.py"
"$ROOT/scripts/lab.sh" exec "nvidia-smi --query-gpu=name,compute_cap,driver_version --format=csv,noheader | head -1 && echo '--- PYTHON ---' && python3 '$PY'"
