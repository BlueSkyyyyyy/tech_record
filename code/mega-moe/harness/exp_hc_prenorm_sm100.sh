#!/usr/bin/env bash
# P87-1 实验 driver：在 kernel_lab(H100/sm90) 上真机验证 SM100 HyperConnection prenorm GEMM
#   csrc/jit_kernels/impls/sm100_tf32_hc_prenorm_gemm.hpp      （host JIT 包装层）
#   deep_gemm/include/deep_gemm/impls/sm100_tf32_hc_prenorm_gemm.cuh （SM100 kernel）
#   ./harness/exp_hc_prenorm_sm100.sh
# 对应源码 src/hc_prenorm_sm100.cu（nvcc 直接 include 真实头文件 + torch）。
#   Part 1 host num_stages / smem 预算循环（SM90 vs SM100）
#   Part 2 split-K 的 K-block 分解（恰好覆盖/余数/部分 D 求和 == 完整 GEMM）
#   Part 3 get_swizzled_smem_offset vs cute::Swizzle<3,4,3>（真实设备代码）+ 双射
#   Part 4 cast-and-reduce warpgroup 的 warp_reduce_sum<4> + m_idx 映射（真实设备代码）
#   Part 5 SM100 wrapper TMA 描述符构造（2D / 3D split，真 torch 张量）
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

CMD=$(cat <<'EOS'
set -e
DG=${HOME}/github/DeepGEMM
TORCH=/usr/local/lib/python3.12/dist-packages/torch
PYSITE=/usr/local/lib/python3.12/dist-packages
SRC=${HOME}/proj/tech_record/code/mega-moe/src/hc_prenorm_sm100.cu

echo "== 环境坐标 =="
( cd $DG && git -c safe.directory='*' log --oneline -1 )
nvidia-smi --query-gpu=name,compute_cap,driver_version --format=csv,noheader | head -1
sha256sum $DG/csrc/jit_kernels/impls/sm100_tf32_hc_prenorm_gemm.hpp \
          $DG/csrc/jit_kernels/impls/sm90_tf32_hc_prenorm_gemm.hpp \
          $DG/deep_gemm/include/deep_gemm/impls/sm100_tf32_hc_prenorm_gemm.cuh \
          $DG/csrc/apis/hyperconnection.hpp | awk '{n=split($2,a,"/"); print $1, a[n]}'
nvcc --version | tail -1

echo
echo "== 编译（nvcc 直接 include 真实头文件 + torch）=="
nvcc -std=c++20 -O1 -w -arch=sm_90a -x cu \
  -I$DG/csrc -I$DG -I$DG/third-party/deep_jit/include -I$DG/third-party/cutlass/include \
  -I/usr/local/cuda/targets/x86_64-linux/include/cccl -I$DG/deep_gemm/include \
  -I$TORCH/include -I$TORCH/include/torch/csrc/api/include \
  -I$PYSITE/pybind11/include -I/usr/include/python3.12 -I/usr/local/cuda/include \
  "$SRC" -o /tmp/hc_prenorm_sm100 \
  -L$TORCH/lib -L/usr/local/cuda/lib64 -Xlinker -rpath -Xlinker $TORCH/lib \
  -ltorch -ltorch_cuda -ltorch_cpu -lc10 -lc10_cuda -lcudart -lcublas -lcublasLt -lpython3.12 -ldl -lpthread
echo "build OK"

echo
echo "== 主实验 =="
/tmp/hc_prenorm_sm100
EOS
)

"$ROOT/scripts/lab.sh" exec "$CMD"
