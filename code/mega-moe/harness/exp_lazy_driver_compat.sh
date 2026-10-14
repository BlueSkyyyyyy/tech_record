#!/usr/bin/env bash
# P80-1 实验 driver：DeepEP / DeepGEMM(deep_jit) 的底层地基真机验证（H100/sm90，纯 host）。
#   * DeepEP   csrc/utils/lazy_driver.hpp   （libcuda 懒加载宏 + 23 个符号）
#   * DeepEP   csrc/utils/tensor.hpp        （get_shape / get_data_ptr）
#   * DeepGEMM csrc/utils/compatibility.hpp （torch/CUDA 版本静态闸门）
#   * deep_jit utils/lazy.hpp / backend/cuda/driver.hpp / utils/exception.hpp（LazyInit、libdw、VMM 符号）
#   * DeepGEMM csrc/jit_kernels/impls/sm100_locality_domain.hpp（最后一个未覆盖文件的 host wrapper 常量）
#   源码 src/lazy_driver_compat.cu（nvcc 直接 include 真实头文件 + torch）。
#   ./harness/exp_lazy_driver_compat.sh > harness/exp_lazy_driver_compat.out.txt
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

CMD=$(cat <<'EOS'
set -e
DEP=${HOME}/github/DeepEP
DG=${HOME}/github/DeepGEMM
TORCH=/usr/local/lib/python3.12/dist-packages/torch
PYSITE=/usr/local/lib/python3.12/dist-packages
SRC=${HOME}/proj/tech_record/code/mega-moe/src/lazy_driver_compat.cu

echo "== 环境坐标 =="
( cd $DEP && git -c safe.directory='*' log --oneline -1 )
( cd $DG && git -c safe.directory='*' log --oneline -1 )
nvidia-smi --query-gpu=name,compute_cap,driver_version --format=csv,noheader | head -1
nvcc --version | tail -1
sha256sum $DEP/csrc/utils/lazy_driver.hpp $DEP/csrc/utils/tensor.hpp \
          $DG/csrc/utils/compatibility.hpp \
          $DG/third-party/deep_jit/include/deep_jit/utils/lazy.hpp \
          $DG/third-party/deep_jit/include/deep_jit/backend/cuda/driver.hpp \
          $DG/third-party/deep_jit/include/deep_jit/utils/exception.hpp \
          $DG/csrc/jit_kernels/impls/sm100_locality_domain.hpp | sed 's#.*/##'

echo
echo "== 编译（nvcc 直接 include 真实头文件 + torch）=="
nvcc -std=c++20 -O1 -g -w -arch=sm_90a -x cu \
  -I$DEP/csrc -I$DEP/deep_ep/include \
  -I$DG/csrc -I$DG -I$DG/deep_gemm/include -I$DG/third-party/deep_jit/include \
  -I$TORCH/include -I$TORCH/include/torch/csrc/api/include \
  -I$PYSITE/pybind11/include -I/usr/include/python3.12 -I/usr/local/cuda/include \
  "$SRC" -o /tmp/lazy_driver_compat \
  -L$TORCH/lib -L/usr/local/cuda/lib64 -Xlinker -rpath -Xlinker $TORCH/lib \
  -ltorch -ltorch_cuda -ltorch_cpu -lc10 -lc10_cuda -lcudart -lcuda -lpython3.12 -ldl -lpthread
echo "build OK"

echo
echo "== 主实验 =="
/tmp/lazy_driver_compat

echo
echo "== 负面用例：版本闸门真的会拦下低版本（编译应失败）=="
cat > /tmp/ver_gate_neg.cu <<'EOF'
#include <cuda.h>
#undef CUDA_VERSION
#define CUDA_VERSION 11000
#include "utils/compatibility.hpp"
int main() { return 0; }
EOF
if nvcc -std=c++20 -w -arch=sm_90a -x cu \
     -I$DG/csrc -I$DG/deep_gemm/include \
     -I$TORCH/include -I$TORCH/include/torch/csrc/api/include \
     -I$PYSITE/pybind11/include -I/usr/include/python3.12 -I/usr/local/cuda/include \
     /tmp/ver_gate_neg.cu -o /tmp/ver_gate_neg 2> /tmp/ver_gate_neg.err; then
  echo "  [FAIL] 伪造 CUDA_VERSION=11000 竟然编译通过（闸门失效）"
else
  echo "  [PASS] 伪造 CUDA_VERSION=11000 被 static_assert 拦下；错误信息："
  grep -m1 -i "requires CUDA Driver API" /tmp/ver_gate_neg.err | sed 's/^ */    /'
fi
EOS
)

"$ROOT/scripts/lab.sh" exec "$CMD"
