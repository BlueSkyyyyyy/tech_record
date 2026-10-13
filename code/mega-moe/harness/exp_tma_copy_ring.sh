#!/usr/bin/env bash
# P67-1 实验 driver：在 kernel_lab(H100/sm90) 上真机验证 DeepGEMM 的共享 TMA 拷贝层与环形流水线
#   deep_gemm/include/deep_gemm/common/tma_copy.cuh + common/ring_pipeline.cuh。
#   ./harness/exp_tma_copy_ring.sh
# 对应源码 src/tma_copy_ring.cu（nvcc 直接 include 真实头文件跑真实设备代码）。
#   Part 1 host constexpr：get_inner_block_atom_size / get_smem_pack_factor（含 FP4 打包）
#   Part 2 设备端 RingPipeline::advance 状态机 vs oracle（含非 2 的幂 / step<=N）
#   Part 3 真实 tma::copy：2D 单原子 / 多原子 atom 循环 / 3D 路径逐字节对比
#   Part 4 真实 tma::copy 多播（cluster=2，LoadMulticast）
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

CMD=$(cat <<'EOS'
set -e
DG=~/github/DeepGEMM
SRC=~/proj/tech_record/code/mega-moe/src/tma_copy_ring.cu

echo "== 环境坐标 =="
git -C $DG -c safe.directory='*' log --oneline -1
nvidia-smi --query-gpu=name,compute_cap,driver_version --format=csv,noheader | head -1
sha256sum $DG/deep_gemm/include/deep_gemm/common/tma_copy.cuh \
          $DG/deep_gemm/include/deep_gemm/common/ring_pipeline.cuh \
          $DG/deep_gemm/include/deep_gemm/common/packing.cuh | sed 's#/home/[^ ]*/#.../#'
nvcc --version | tail -1

echo
echo "== 编译（nvcc 直接 include 真实 tma_copy.cuh / ring_pipeline.cuh）=="
nvcc -std=c++20 -O2 -w -arch=sm_90a --expt-relaxed-constexpr -x cu \
  -I$DG/deep_gemm/include -I$DG/third-party/cutlass/include \
  -I/usr/local/cuda/targets/x86_64-linux/include/cccl -I/usr/local/cuda/include \
  "$SRC" -o /tmp/tma_copy_ring \
  -L/usr/local/cuda/lib64 -lcudart -lcuda
echo "build OK"

echo
echo "== 主实验 =="
/tmp/tma_copy_ring
EOS
)

"$ROOT/scripts/lab.sh" exec "$CMD"
