#!/usr/bin/env bash
# P172-1 实验 driver：在 kernel_lab(H100) 上运行 harness/exp_moe_recompute_overlap.py，
# 用真实 SM90 FP8 grouped kernel 量「重算 L1 ∥ 不依赖 h 的反向 GEMM」的并发重叠 vs SM 预算分区。
#   ./harness/exp_moe_recompute_overlap.sh
# 输出见 harness/exp_moe_recompute_overlap.out.txt。
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="$ROOT/harness/exp_moe_recompute_overlap.py"
OUT="$ROOT/harness/exp_moe_recompute_overlap.out.txt"
GPU="${CUDA_VISIBLE_DEVICES:-1}"
"$ROOT/scripts/lab.sh" exec "CUDA_VISIBLE_DEVICES=$GPU nvidia-smi --query-gpu=name,compute_cap,driver_version --format=csv,noheader | head -1; echo '--- PYTHON (GPU='$GPU') ---'; CUDA_VISIBLE_DEVICES=$GPU python3 '$PY'" | tee "$OUT"
