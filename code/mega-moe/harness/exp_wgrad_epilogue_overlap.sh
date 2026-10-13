#!/usr/bin/env bash
# P138-1 实验 driver：MoE wgrad 的 S 步归约能否「藏进 GEMM 的 HBM 余量」。
#   bash harness/exp_wgrad_epilogue_overlap.sh > harness/exp_wgrad_epilogue_overlap.out.txt
#   段一：nvcc 编 src/wgrad_epilogue_overlap.cu（(a) 独立可执行跑自带 [A]/[B] 报告；
#         (b) 编 .so 由 Python 经 ctypes 驱动其消费侧内核）。
#   段二：harness/exp_wgrad_epilogue_overlap.py —— 真实 SM90 GEMM + 串行/重叠四部分。
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
H="$ROOT/harness"
SRC="$ROOT/src/wgrad_epilogue_overlap.cu"

CMD=$(cat <<EOS
set -e
echo "== 内核源码 =="
sha256sum '$SRC'
echo
echo "== 编译（sm_90）=="
nvcc -arch=sm_90 -O3 -std=c++17 '$SRC' -o /tmp/wgrad_epilogue_overlap
nvcc -arch=sm_90 -O3 -std=c++17 -DWOC_NO_MAIN -shared -Xcompiler -fPIC '$SRC' -o '$H/wgrad_epilogue_overlap.so'
echo "  built /tmp/wgrad_epilogue_overlap and $H/wgrad_epilogue_overlap.so"
echo
echo "== 设备端自带报告（src/wgrad_epilogue_overlap.cu，单进程）=="
nvidia-smi --query-gpu=name,compute_cap,driver_version --format=csv,noheader | head -1
CUDA_VISIBLE_DEVICES=\${CUDA_VISIBLE_DEVICES:-1} /tmp/wgrad_epilogue_overlap
echo
echo "== Python：真实 GEMM + 串行/重叠（空闲 GPU）=="
cd '$H'
CUDA_VISIBLE_DEVICES=\${CUDA_VISIBLE_DEVICES:-1} python3 -u exp_wgrad_epilogue_overlap.py
EOS
)

"$ROOT/scripts/lab.sh" exec "$CMD"
