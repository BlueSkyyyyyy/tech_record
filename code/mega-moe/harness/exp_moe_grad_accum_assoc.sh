#!/usr/bin/env bash
# P174-1 实验 driver：H100(sm90) 上量化 MoE 反向梯度累积的「切分 × 结合序」可复现性。
#   bash harness/exp_moe_grad_accum_assoc.sh > harness/exp_moe_grad_accum_assoc.out.txt
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
H="$ROOT/harness"
"$ROOT/scripts/lab.sh" exec "export CUDA_VISIBLE_DEVICES=\${CUDA_VISIBLE_DEVICES:-3}; cd '$H' && \
nvidia-smi --query-gpu=name,compute_cap,driver_version --format=csv,noheader | head -1 && \
python exp_moe_grad_accum_assoc.py"
