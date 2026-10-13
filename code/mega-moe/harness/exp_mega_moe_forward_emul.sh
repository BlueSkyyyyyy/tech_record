#!/usr/bin/env bash
# P89-1 实验 driver：在 kernel_lab(H100) 上运行 harness/exp_mega_moe_forward_emul.py，
# 用 SM90 grouped FP8/BF16 kernel 复刻 Mega MoE 前向流水线并做误差分解。
#   ./harness/exp_mega_moe_forward_emul.sh
# 输出见 harness/exp_mega_moe_forward_emul.out.txt。
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="$ROOT/harness/exp_mega_moe_forward_emul.py"
"$ROOT/scripts/lab.sh" exec "nvidia-smi --query-gpu=name,compute_cap,driver_version --format=csv,noheader | head -1 && echo '--- PYTHON ---' && python3 '$PY'"
