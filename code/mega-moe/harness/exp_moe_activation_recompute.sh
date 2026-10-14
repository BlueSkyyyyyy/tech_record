#!/usr/bin/env bash
# P171-1 实验 driver：在 kernel_lab(H100) 上运行 harness/exp_moe_activation_recompute.py，
# 用真实 SM90 FP8 grouped/dense kernel 量 MoE 反向「激活存储 vs 重算」的逐位性 / 显存 / 时间。
#   ./harness/exp_moe_activation_recompute.sh
# 输出见 harness/exp_moe_activation_recompute.out.txt。
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="$ROOT/harness/exp_moe_activation_recompute.py"
OUT="$ROOT/harness/exp_moe_activation_recompute.out.txt"
GPU="${CUDA_VISIBLE_DEVICES:-1}"
"$ROOT/scripts/lab.sh" exec "CUDA_VISIBLE_DEVICES=$GPU nvidia-smi --query-gpu=name,compute_cap,driver_version --format=csv,noheader | head -1; echo '--- PYTHON (GPU='$GPU') ---'; CUDA_VISIBLE_DEVICES=$GPU python3 '$PY'" | tee "$OUT"
