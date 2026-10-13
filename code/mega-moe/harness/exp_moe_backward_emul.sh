#!/usr/bin/env bash
# P101-1 实验 driver：在 kernel_lab(H100) 上运行 harness/exp_moe_backward_emul.py，
# 用真实参考量化器 + SM90 grouped FP8 kernel 复刻 MoE 反向并做误差来源分解（P89-1 前向复刻的对偶）。
#   ./harness/exp_moe_backward_emul.sh
# 输出见 harness/exp_moe_backward_emul.out.txt。
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="$ROOT/harness/exp_moe_backward_emul.py"
"$ROOT/scripts/lab.sh" exec "nvidia-smi --query-gpu=name,compute_cap,driver_version --format=csv,noheader | head -1 && echo '--- PYTHON ---' && python3 '$PY'"
