#!/usr/bin/env bash
# P79-1 实验 driver：在 kernel_lab(H100/sm90) 上真机验证两个 Mega 家族 JIT 包装层
#   csrc/jit_kernels/impls/sm100_bf16_mega_moe.hpp  （P10-1 拆过 kernel，本篇补 wrapper）
#   csrc/jit_kernels/impls/sm100_mega_mhc.hpp       （P5-2 拆过 kernel，本篇补 wrapper）
#   ./harness/exp_mega_moe_mhc_jit_wrapper.sh
# 对应源码 src/mega_moe_mhc_jit_wrapper.cu（nvcc 直接 include 真实头文件 + torch）。
#   Part 1 to_string(float) 十六进制字面量往返
#   Part 2 MixArgs / NormArgs 的 host<->device ABI 布局
#   Part 3 mHC wrapper 的三个 TMA lambda + 真实 3D TMA 载入（route 当第三维）
#   Part 4 BF16 Mega MoE wrapper 描述符推导与 FP8 差异 + 共享专家回退
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

CMD=$(cat <<'EOS'
set -e
DG=${HOME}/github/DeepGEMM
TORCH=/usr/local/lib/python3.12/dist-packages/torch
PYSITE=/usr/local/lib/python3.12/dist-packages
SRC=${HOME}/proj/tech_record/code/mega-moe/src/mega_moe_mhc_jit_wrapper.cu

echo "== 环境坐标 =="
( cd $DG && git -c safe.directory='*' log --oneline -1 )
nvidia-smi --query-gpu=name,compute_cap,driver_version --format=csv,noheader | head -1
sha256sum $DG/csrc/jit_kernels/impls/sm100_bf16_mega_moe.hpp \
          $DG/csrc/jit_kernels/impls/sm100_mega_mhc.hpp \
          $DG/csrc/jit_kernels/impls/runtime_utils.hpp \
          $DG/deep_gemm/include/deep_gemm/layout/mega_mhc.cuh | sed 's#.*/##'
nvcc --version | tail -1

echo
echo "== 编译（nvcc 直接 include 真实头文件 + torch）=="
nvcc -std=c++20 -O1 -w -arch=sm_90a -x cu \
  -I$DG/csrc -I$DG -I$DG/third-party/deep_jit/include -I$DG/third-party/cutlass/include \
  -I/usr/local/cuda/targets/x86_64-linux/include/cccl -I$DG/deep_gemm/include \
  -I$TORCH/include -I$TORCH/include/torch/csrc/api/include \
  -I$PYSITE/pybind11/include -I/usr/include/python3.12 -I/usr/local/cuda/include \
  "$SRC" -o /tmp/mega_moe_mhc_jit_wrapper \
  -L$TORCH/lib -L/usr/local/cuda/lib64 -Xlinker -rpath -Xlinker $TORCH/lib \
  -ltorch -ltorch_cuda -ltorch_cpu -lc10 -lc10_cuda -lcudart -lcublas -lcublasLt -lpython3.12 -ldl -lpthread
echo "build OK"

echo
echo "== 主实验 =="
/tmp/mega_moe_mhc_jit_wrapper
EOS
)

"$ROOT/scripts/lab.sh" exec "$CMD"
