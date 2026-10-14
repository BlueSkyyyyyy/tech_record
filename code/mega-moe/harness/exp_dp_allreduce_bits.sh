#!/usr/bin/env bash
# P175-1 实验 driver：H100 上真实 NCCL 的 DP 梯度 all-reduce 逐位一致性。
#   bash harness/exp_dp_allreduce_bits.sh > harness/exp_dp_allreduce_bits.out.txt
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
H="$ROOT/harness"
"$ROOT/scripts/lab.sh" exec "export CUDA_VISIBLE_DEVICES=\${CUDA_VISIBLE_DEVICES:-1,2,3,4}; export OMP_NUM_THREADS=4; cd '$H' && \
nvidia-smi --query-gpu=name,compute_cap,driver_version --format=csv,noheader | head -1 && \
python exp_dp_allreduce_bits.py"
