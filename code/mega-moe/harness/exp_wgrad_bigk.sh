#!/usr/bin/env bash
# P140-1 实验 driver：把 MoE wgrad 的 S 步梯度累积「折进收缩维 K」（big-K 折叠）能否消掉每步累积税。
#   bash harness/exp_wgrad_bigk.sh > harness/exp_wgrad_bigk.out.txt
#   第一段：exp_wgrad_bigk.py —— Part A 正确性 / Part B 端到端计时 / Part C 字节账与前提代价。
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
H="$ROOT/harness"
"$ROOT/scripts/lab.sh" exec "export CUDA_VISIBLE_DEVICES=\${CUDA_VISIBLE_DEVICES:-1}; cd '$H' && \
nvidia-smi --query-gpu=name,compute_cap,driver_version --format=csv,noheader | head -1 && \
python3 -u exp_wgrad_bigk.py"
