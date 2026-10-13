#!/usr/bin/env bash
# P145-1 实验 driver：把反向的 SwiGLU 段折进 dgrad 的 epilogue。
#   bash harness/exp_moe_backward_fused.sh > harness/exp_moe_backward_fused.out.txt
#   段一：nvcc 编 src/moe_backward_activation_fused.cu（输出侧原型：正确性 [A]、算力扫描 [B]）。
#   段二：harness/exp_moe_backward_fused.py（字节账 + 真实 d2g anchor + 预算组装）。
#   段三：ncu 单次 launch 采真实 d2g 的 DRAM 读/写（坐实「有没有带宽余量」）。
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
H="$ROOT/harness"
SRC="$ROOT/src/moe_backward_activation_fused.cu"

CMD=$(cat <<EOS
set -e
echo "== 内核源码 =="
sha256sum '$SRC'
echo
echo "== 编译（sm_90）=="
nvcc -arch=sm_90 -O3 -std=c++17 '$SRC' -o /tmp/moe_backward_activation_fused
echo "  built /tmp/moe_backward_activation_fused"
echo
echo "== 设备端自带报告（src/moe_backward_activation_fused.cu，单进程，空闲 GPU）=="
nvidia-smi --query-gpu=name,compute_cap,driver_version --format=csv,noheader | head -1
CUDA_VISIBLE_DEVICES=\${CUDA_VISIBLE_DEVICES:-1} /tmp/moe_backward_activation_fused
echo
echo "== Python：字节账 + 真实 d2g anchor + 预算组装（空闲 GPU）=="
cd '$H'
CUDA_VISIBLE_DEVICES=\${CUDA_VISIBLE_DEVICES:-1} python3 -u exp_moe_backward_fused.py
echo
echo "== ncu：真实 d2g 单次 launch 的 DRAM 读/写（带宽余量）=="
KNAME="--target-processes all --launch-count 1 --kernel-name regex:sm90_fp8_gemm"
METRICS="gpu__time_duration.sum,dram__bytes_read.sum,dram__bytes_write.sum,sm__inst_executed_pipe_tensor_op_gmma.sum"
SKIP_EVENT_PRINT=1 CUDA_VISIBLE_DEVICES=\${CUDA_VISIBLE_DEVICES:-1} ncu \$KNAME --metrics \$METRICS python exp_backward_ncu.py d2g 2>&1 | grep -E 'gpu__time|dram__bytes|tensor_op_gmma'
EOS
)

"$ROOT/scripts/lab.sh" exec "$CMD"
