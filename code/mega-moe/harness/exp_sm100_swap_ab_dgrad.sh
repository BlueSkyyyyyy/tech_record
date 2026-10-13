#!/usr/bin/env bash
# P93-1 实验 driver：在 kernel_lab(H100/sm90) 上真机验证 SM100 BF16 GEMM 的 `kSwapAB`(dgrad) 分支的
#   **host 半边**（选核启发式 / storage / pipeline / launch），以及「swap == transpose」的数值恒等式。
#   主 kernel（UMMA/TMEM/转置 STSM）是 SM100 专属，仅分析（见 docs/103）。
# 对应源码 src/sm100_swap_ab_dgrad.cu（nvcc 直接 include 真实头文件 + torch）。
#   Part A  m-grouped -> 唯一 swap_ab layout（block_n==128, cluster_m==1, block_m=mk_alignment）
#   Part A2 block_m 随 mk_alignment(128/256) 变；storage 派生
#   Part A3 mask / psum m-grouped 同样强制 swap_ab
#   Part B  storage / pipeline / launch 逐字段对回源码公式
#   Part B2 mk=256 的 smem 预算（更少 stage）
#   Part B3 稠密 GEMM 枚举 swap_ab∈{0,1}（对照）
#   Part C  (A@B^T)^T == B@A^T 逐位（真机 torch，含逐组）
#   Part D  SM90 无 swap_ab（对照）
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

CMD=$(cat <<'EOS'
set -e
DG=~/github/DeepGEMM
TORCH=/usr/local/lib/python3.12/dist-packages/torch
PYSITE=/usr/local/lib/python3.12/dist-packages
SRC=~/proj/tech_record/code/mega-moe/src/sm100_swap_ab_dgrad.cu

echo "== 环境坐标 =="
( cd $DG && git -c safe.directory='*' log --oneline -1 )
nvidia-smi --query-gpu=name,compute_cap,driver_version --format=csv,noheader | head -1
sha256sum $DG/csrc/jit_kernels/impls/sm100_bf16_gemm.hpp \
          $DG/deep_gemm/include/deep_gemm/impls/sm100_bf16_gemm.cuh \
          $DG/deep_gemm/include/deep_gemm/epilogue/sm100_store_cd_swap_ab.cuh \
          $DG/csrc/jit_kernels/heuristics/sm100.hpp \
          $DG/csrc/jit_kernels/heuristics/sm90.hpp | awk '{n=split($2,a,"/"); print $1, a[n]}'
nvcc --version | tail -1

echo
echo "== 编译（nvcc 直接 include 真实头文件 + torch）=="
nvcc -std=c++20 -O1 -w -arch=sm_90a -x cu \
  -I$DG/csrc -I$DG -I$DG/third-party/deep_jit/include -I$DG/third-party/cutlass/include \
  -I/usr/local/cuda/targets/x86_64-linux/include/cccl -I$DG/deep_gemm/include \
  -I$TORCH/include -I$TORCH/include/torch/csrc/api/include \
  -I$PYSITE/pybind11/include -I/usr/include/python3.12 -I/usr/local/cuda/include \
  "$SRC" -o /tmp/sm100_swap_ab_dgrad \
  -L$TORCH/lib -L/usr/local/cuda/lib64 -Xlinker -rpath -Xlinker $TORCH/lib \
  -ltorch -ltorch_cuda -ltorch_cpu -lc10 -lc10_cuda -lcudart -lcublas -lcublasLt -lpython3.12 -ldl -lpthread
echo "build OK"

echo
echo "== 主实验 =="
/tmp/sm100_swap_ab_dgrad
EOS
)

"$ROOT/scripts/lab.sh" exec "$CMD"
