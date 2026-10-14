#!/usr/bin/env bash
# P133-1 实验 driver：在 kernel_lab(H100/sm90) 上复刻 Mega MoE **前向收尾两段**
#   （L2 bf16 epilogue 写回 + combine 归约），补全第 142 篇缺的「非 GEMM 段」。
#   bash harness/exp_moe_forward_combine.sh > harness/exp_moe_forward_combine.out.txt
# 对应源码 src/moe_forward_combine.cu（直接 include 真实头文件跑真实设备代码）与
#   harness/exp_moe_forward_combine.py（独立 torch oracle + 8×H100 远端邮箱带宽）。
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
H="$ROOT/harness"
INC="${HOME}/github/DeepGEMM/deep_gemm/include"

CMD=$(cat <<EOS
set -e
echo "== 真实头文件 sha256 =="
sha256sum $INC/deep_gemm/common/math.cuh $INC/deep_gemm/ptx/utils.cuh
echo
echo "== 设备端：L2 写回 + combine 归约（src/moe_forward_combine.cu，sm_90）=="
nvcc -arch=sm_90 -O3 -std=c++17 --expt-relaxed-constexpr -I"$INC" -I"$INC/cutlass" -I"$INC/cute" '$ROOT/src/moe_forward_combine.cu' -o /tmp/moe_forward_combine
nvidia-smi --query-gpu=name,compute_cap,driver_version --format=csv,noheader | head -1
/tmp/moe_forward_combine
echo
echo "== Python 侧：独立 oracle =="
cd '$H'
python3 exp_moe_forward_combine.py
echo
echo "== Python 侧：远端邮箱搬运带宽（2 卡 NCCL，独立进程，选空闲的 GPU）=="
CUDA_VISIBLE_DEVICES=\${MOE_COMM_GPUS:-2,3} NCCL_NVLS_ENABLE=0 PYTORCH_NO_CUDA_MEMORY_CACHING=0 torchrun --nproc_per_node=2 exp_moe_forward_combine.py --dist 2>&1 | grep -E "^(A|B)\)" | tail -6
EOS
)

"$ROOT/scripts/lab.sh" exec "$CMD"
