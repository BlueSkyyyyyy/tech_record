#!/usr/bin/env bash
# P146-1 实验 driver：把反向的 SwiGLU 段落地成 SM100 dgrad 的 epilogue_class operator。
#   bash harness/exp_moe_backward_activation_epilogue.sh > harness/exp_moe_backward_activation_epilogue.out.txt
#   段一：nvcc 编 src/moe_backward_activation_epilogue.cu（契约尺寸账 [A]、真实 math.cuh store 分支 vs 手写参考 [B]、
#         单输出 vs 双输出 [C/C2]）。
#   段二：harness/exp_moe_backward_activation_epilogue.py（host 接线：类型字符串映射 / JIT 注入 / resolve 互斥）。
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
H="$ROOT/harness"
SRC="$ROOT/src/moe_backward_activation_epilogue.cu"
DG="${DEEPGEMM_DIR:-~/github/DeepGEMM}"
INC="$DG/deep_gemm/include"

CMD=$(cat <<EOS
set -e
echo "== 内核源码 =="
sha256sum '$SRC'
echo
echo "== 编译（sm_90；包含真实 DeepGEMM 头 types.cuh / common/math.cuh）=="
nvcc -arch=sm_90 -O3 -std=c++17 -I'$INC' -I'$INC/cute' -I'$INC/cutlass' \
  '$SRC' -o /tmp/moe_backward_activation_epilogue
echo "  built /tmp/moe_backward_activation_epilogue"
echo
echo "== 设备端自带报告（单进程，空闲 GPU）=="
nvidia-smi --query-gpu=name,compute_cap,driver_version --format=csv,noheader | head -1
CUDA_VISIBLE_DEVICES=\${CUDA_VISIBLE_DEVICES:-1} /tmp/moe_backward_activation_epilogue
echo
echo "== Python：host 接线与契约缺口 =="
cd '$H'
DEEPGEMM_DIR='$DG' CUDA_VISIBLE_DEVICES=\${CUDA_VISIBLE_DEVICES:-1} python3 -u exp_moe_backward_activation_epilogue.py
EOS
)

"$ROOT/scripts/lab.sh" exec "$CMD"
