#!/usr/bin/env bash
# P159-1 实验 driver：在 kernel_lab(H100) 上运行 harness/exp_moe_train_step.py，
# 用真实 SM90 FP8 kernel 把「一个 MoE 训练步（前向+反向）」分段搭起来并计时，
# 并闭环共享专家（单组 grouped GEMM）的反向。
#   ./harness/exp_moe_train_step.sh
# 输出见 harness/exp_moe_train_step.out.txt。
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="$ROOT/harness/exp_moe_train_step.py"
"$ROOT/scripts/lab.sh" exec "nvidia-smi --query-gpu=name,compute_cap,driver_version --format=csv,noheader | head -1 && echo '--- PYTHON ---' && python3 '$PY'"
