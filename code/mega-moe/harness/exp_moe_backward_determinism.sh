#!/usr/bin/env bash
# P170-1 实验 driver：H100(sm90) 上量化 MoE 反向 grouped GEMM 的可复现性边界。
#   bash harness/exp_moe_backward_determinism.sh > harness/exp_moe_backward_determinism.out.txt
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
H="$ROOT/harness"
"$ROOT/scripts/lab.sh" exec "export CUDA_VISIBLE_DEVICES=\${CUDA_VISIBLE_DEVICES:-3}; cd '$H' && \
nvidia-smi --query-gpu=name,compute_cap,driver_version --format=csv,noheader | head -1 && \
python exp_moe_backward_determinism.py"
