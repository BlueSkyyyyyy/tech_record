#!/usr/bin/env bash
# P125-1 实验 driver：在 kernel_lab(H100/sm90) 上直接编译并运行
#   src/mhc_backward_segment.cu —— 用**真实头文件**核对 mHC 反向单 kernel 接线的关键一环：
#     A) 前向 Post worker 的 lane-ownership 网格（由 layout/mega_mhc.cuh 派生）
#     B) device 分段归约 vs host double（正确网格 vs 全 32-lane 错误归约）
#     C) TMEM 列预算（utils::get_num_aligned_tmem_cols）
#     D) 分段网格 vs 朴素 worker 的真实吞吐
#   + harness/exp_mhc_backward_segment.py —— 真实形状（H=7168）的目标卡接线账目。
#   ./harness/exp_mhc_backward_segment.sh
# 输出见 harness/exp_mhc_backward_segment.out.txt。
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC="$ROOT/src/mhc_backward_segment.cu"
PY="$ROOT/harness/exp_mhc_backward_segment.py"
INC=~/github/DeepGEMM/deep_gemm/include
OUTDIR=~/tmp
BIN=$OUTDIR/mhc_backward_segment

"$ROOT/scripts/lab.sh" exec "
set -e
echo '== 真实头文件 sha256 =='
sha256sum $INC/deep_gemm/layout/mega_mhc.cuh $INC/deep_gemm/common/math.cuh $INC/deep_gemm/common/utils.cuh
echo '== 编译（nvcc -arch=sm_90，真实头文件）=='
nvcc -arch=sm_90 -O3 -std=c++17 --expt-relaxed-constexpr -diag-suppress 2389 \
  -I$INC -I$INC/cutlass -I$INC/cute '$SRC' -o $BIN 2>&1 | grep -E 'error|warning: ' || true
echo '== 运行 =='
$BIN
echo
echo '== Python 侧：真实形状（H=7168）目标卡接线账目 =='
python3 '$PY'
"
