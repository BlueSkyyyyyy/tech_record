#!/usr/bin/env bash
# P144-1 实验 driver：MoE 反向「非 GEMM 段」——SwiGLU 反向 + FP8/UE8M0 重量化的下界内核与预算。
#   bash harness/exp_moe_backward_activation.sh > harness/exp_moe_backward_activation.out.txt
#   段一：nvcc 编 src/moe_backward_activation.cu（真实内核：正确性 [A]、bf16-round 误差 [B]、带宽 [C]）。
#   段二：harness/exp_moe_backward_activation.py —— 误差消融（torch fp64 参考）+ 反向 step 预算组装。
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
H="$ROOT/harness"
SRC="$ROOT/src/moe_backward_activation.cu"

CMD=$(cat <<EOS
set -e
echo "== 内核源码 =="
sha256sum '$SRC'
echo
echo "== 编译（sm_90）=="
nvcc -arch=sm_90 -O3 -std=c++17 '$SRC' -o /tmp/moe_backward_activation
echo "  built /tmp/moe_backward_activation"
echo
echo "== 设备端自带报告（src/moe_backward_activation.cu，单进程，空闲 GPU）=="
nvidia-smi --query-gpu=name,compute_cap,driver_version --format=csv,noheader | head -1
CUDA_VISIBLE_DEVICES=\${CUDA_VISIBLE_DEVICES:-1} /tmp/moe_backward_activation
echo
echo "== Python：误差消融 + 反向 step 预算（空闲 GPU）=="
cd '$H'
CUDA_VISIBLE_DEVICES=\${CUDA_VISIBLE_DEVICES:-1} python3 -u exp_moe_backward_activation.py
EOS
)

"$ROOT/scripts/lab.sh" exec "$CMD"
