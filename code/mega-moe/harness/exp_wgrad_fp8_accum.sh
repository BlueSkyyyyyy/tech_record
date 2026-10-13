#!/usr/bin/env bash
# P141-1 实验 driver：MoE wgrad 的「FP8 累加器」路线——把每步累积器 RMW 从 8B 压到 2B 的
#   误差-字节边界（与第 150 篇的 big-K 折叠对比）。
#   bash harness/exp_wgrad_fp8_accum.sh > harness/exp_wgrad_fp8_accum.out.txt
#   段一：nvcc 编 src/fp8_accum_rmw.cu（真实内核：fp32/bf16/fp8 累积器 RMW 带宽 + 数值自检）。
#   段二：harness/exp_wgrad_fp8_accum.py —— Part A（隔离误差随 S）/ Part B（真实 SM90 kernel）
#         / Part C（字节账与模型时间）。
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
H="$ROOT/harness"
SRC="$ROOT/src/fp8_accum_rmw.cu"

CMD=$(cat <<EOS
set -e
echo "== 内核源码 =="
sha256sum '$SRC'
echo
echo "== 编译（sm_90）=="
nvcc -arch=sm_90 -O3 -std=c++17 '$SRC' -o /tmp/fp8_accum_rmw
echo "  built /tmp/fp8_accum_rmw"
echo
echo "== 设备端自带报告（src/fp8_accum_rmw.cu，单进程）=="
nvidia-smi --query-gpu=name,compute_cap,driver_version --format=csv,noheader | head -1
CUDA_VISIBLE_DEVICES=\${CUDA_VISIBLE_DEVICES:-1} /tmp/fp8_accum_rmw
echo
echo "== Python：Part A/B/C（空闲 GPU）=="
cd '$H'
CUDA_VISIBLE_DEVICES=\${CUDA_VISIBLE_DEVICES:-1} python3 -u exp_wgrad_fp8_accum.py
EOS
)

"$ROOT/scripts/lab.sh" exec "$CMD"
