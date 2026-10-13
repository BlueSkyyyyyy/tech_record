#!/usr/bin/env bash
# P137-1 实验 driver：量化 MoE wgrad 的「fp32 物化 / 归约税」能拉回多少 2.2×。
#   bash harness/exp_wgrad_reduce_consumer.sh > harness/exp_wgrad_reduce_consumer.out.txt
#   段一：nvcc 编 src/wgrad_reduce_consumer.cu（(a) 独立可执行跑自带 [A]/[B] 报告；
#         (b) 编 .so 由 Python 经 ctypes 驱动其消费侧内核）。
#   段二：harness/exp_wgrad_reduce_consumer.py —— 真实 SM90 GEMM + 三协议端到端。
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
H="$ROOT/harness"
SRC="$ROOT/src/wgrad_reduce_consumer.cu"

CMD=$(cat <<EOS
set -e
echo "== 内核源码 =="
sha256sum '$SRC'
echo
echo "== 编译（sm_90）=="
nvcc -arch=sm_90 -O3 -std=c++17 '$SRC' -o /tmp/wgrad_reduce_consumer
nvcc -arch=sm_90 -O3 -std=c++17 -DWRC_NO_MAIN -shared -Xcompiler -fPIC '$SRC' -o '$H/wgrad_reduce_consumer.so'
echo "  built /tmp/wgrad_reduce_consumer and $H/wgrad_reduce_consumer.so"
echo
echo "== 设备端自带报告（src/wgrad_reduce_consumer.cu，单进程）=="
nvidia-smi --query-gpu=name,compute_cap,driver_version --format=csv,noheader | head -1
CUDA_VISIBLE_DEVICES=\${CUDA_VISIBLE_DEVICES:-1} /tmp/wgrad_reduce_consumer
echo
echo "== Python：真实 GEMM + 三协议端到端（空闲 GPU1）=="
cd '$H'
CUDA_VISIBLE_DEVICES=\${CUDA_VISIBLE_DEVICES:-1} python3 exp_wgrad_reduce_consumer.py
EOS
)

"$ROOT/scripts/lab.sh" exec "$CMD"
