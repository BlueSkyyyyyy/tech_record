#!/usr/bin/env bash
# P136-1 实验 driver：在 kernel_lab(H100) 量化 MoE wgrad 换成 bf16 输出的精度代价与落地。
#   bash harness/exp_wgrad_bf16_output.sh > harness/exp_wgrad_bf16_output.out.txt
#   第一段：exp_wgrad_bf16_output.py —— 精度分解（Part A）、跨步累积（Part B）、计时（Part C）。
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
H="$ROOT/harness"
"$ROOT/scripts/lab.sh" exec "export CUDA_VISIBLE_DEVICES=\${CUDA_VISIBLE_DEVICES:-1}; cd '$H' && \
nvidia-smi --query-gpu=name,compute_cap,driver_version --format=csv,noheader | head -1 && \
python exp_wgrad_bf16_output.py"
