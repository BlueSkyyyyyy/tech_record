#!/usr/bin/env bash
# P46-1 实验 driver：在 kernel_lab(H100/sm90) 上真机验证 DeepGEMM 的 TMA 描述符构造层
#   csrc/jit_kernels/impls/runtime_utils.hpp（make_tma_* + dtype/swizzle 映射）。
#   ./harness/exp_tma_descriptor.sh
# 对应源码 src/tma_descriptor.cu（nvcc 直接 include 真实 runtime_utils.hpp + torch）。
#   Part 1 wrapper vs 直连 cuTensorMapEncodeTiled 逐字节等价
#   Part 2 真实 cp.async.bulk.tensor.2d 载入 vs cute::Swizzle<B,4,3> 预测 + swizzle 地址对齐陷阱
#   Part 3 128 字节描述符指纹（反推字段布局）
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

CMD=$(cat <<'EOS'
set -e
DG=${HOME}/github/DeepGEMM
TORCH=/usr/local/lib/python3.12/dist-packages/torch
PYSITE=/usr/local/lib/python3.12/dist-packages
SRC=${HOME}/proj/tech_record/code/mega-moe/src/tma_descriptor.cu

echo "== 环境坐标 =="
( cd $DG && git -c safe.directory='*' log --oneline -1 )
nvidia-smi --query-gpu=name,compute_cap,driver_version --format=csv,noheader | head -1
sha256sum $DG/csrc/jit_kernels/impls/runtime_utils.hpp $DG/csrc/utils/math.hpp \
          $DG/third-party/cutlass/include/cute/swizzle.hpp | sed 's#.*/##'
nvcc --version | tail -1

echo
echo "== 编译（nvcc 直接 include 真实 runtime_utils.hpp + torch）=="
nvcc -std=c++20 -O1 -w -arch=sm_90a -x cu \
  -I$DG/csrc -I$DG -I$DG/third-party/deep_jit/include -I$DG/third-party/cutlass/include \
  -I/usr/local/cuda/targets/x86_64-linux/include/cccl -I$DG/deep_gemm/include \
  -I$TORCH/include -I$TORCH/include/torch/csrc/api/include \
  -I$PYSITE/pybind11/include -I/usr/include/python3.12 -I/usr/local/cuda/include \
  "$SRC" -o /tmp/tma_descriptor \
  -L$TORCH/lib -L/usr/local/cuda/lib64 -Xlinker -rpath -Xlinker $TORCH/lib \
  -ltorch -ltorch_cuda -ltorch_cpu -lc10 -lc10_cuda -lcudart -lcublas -lcublasLt -lpython3.12 -ldl -lpthread
echo "build OK"

echo
echo "== 主实验 =="
/tmp/tma_descriptor
EOS
)

"$ROOT/scripts/lab.sh" exec "$CMD"
