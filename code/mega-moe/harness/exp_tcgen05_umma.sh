#!/usr/bin/env bash
# P100-1 实验 driver：H100(sm90a) 上验证 SM100 UMMA（tcgen05）指令底座与 SM100 `bmk,bnk->mn` 批归约 GEMM
#   的**架构无关部分**。
#   Part A  ptxas 编译矩阵 —— 真实 tcgen05.cuh 的每条指令在 sm_90a 下被拒 / sm_100a 下通过（硬证据）
#   Part 1  host：bmk epilogue 的 STSM swizzle 整数公式 vs cute::Swizzle<3,4,3>
#   Part 2  device：真实 TMA reduce-add（cp.reduce.async.bulk.tensor.2d ... add.f32）累积语义
#   Part 3  device：swizzle-128 的 TMA reduce-add（对齐 bmk kernel D 描述符方向）
#   Part 4  host：真实 ptx/tcgen05.cuh 在 sm90a 下可解析 + 描述符尺寸
# 对应源码 src/tcgen05_umma.cu（nvcc 直接 include 真实头文件）。
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

CMD=$(cat <<'EOS'
set -e
DG=~/github/DeepGEMM
SRC=~/proj/tech_record/code/mega-moe/src/tcgen05_umma.cu
INC="-I$DG/deep_gemm/include -I$DG/third-party/cutlass/include -I$DG/third-party/deep_jit/include -I$DG/csrc"

echo "== 环境坐标 =="
( cd $DG && git -c safe.directory='*' log --oneline -1 )
nvidia-smi --query-gpu=name,compute_cap,driver_version --format=csv,noheader | head -1
nvcc --version | tail -1
sha256sum $DG/deep_gemm/include/deep_gemm/ptx/tcgen05.cuh \
          $DG/deep_gemm/include/deep_gemm/impls/sm100_bmk_bnk_mn.cuh \
          $DG/csrc/jit_kernels/impls/sm100_bmk_bnk_mn.hpp \
          $DG/deep_gemm/include/deep_gemm/mma/sm100.cuh | awk '{n=split($2,a,"/"); print substr($1,1,16), a[n]}'

echo
echo "== Part A: tcgen05 指令编译矩阵（sm_90a 应被 ptxas 拒绝 / sm_100a 应通过）=="
TMP=/tmp/tcgen05_matrix
mkdir -p $TMP
hdr='#include <cute/atom/copy_traits_sm100.hpp>
#include <deep_gemm/ptx/tcgen05.cuh>
#include <deep_gemm/mma/sm100.cuh>
using namespace deep_gemm;
__global__ void k() {'
ftr='}'

emit() { # $1=name  $2=body
  printf '%s\n  %s\n%s\n' "$hdr" "$2" "$ftr" > $TMP/$1.cu
}
emit f16bf16_ss      'uint64_t a=0,b=0,d=0; uint32_t c=0,s=1; ptx::SM100_MMA_F16BF16_SS::fma(a,b,c,s,d);'
emit f16bf16_2x1     'uint64_t a=0,b=0,d=0; uint32_t c=0,s=1; ptx::SM100_MMA_F16BF16_2x1SM_SS::fma(a,b,c,s,d);'
emit mxf8f6f4_ss     'uint64_t a=0,b=0,d=0; uint32_t c=0,s=1,sa=0,sb=0; ptx::SM100_MMA_MXF8F6F4_SS::fma(a,b,c,s,d,sa,sb);'
emit mxf4_ss         'uint64_t a=0,b=0,d=0; uint32_t c=0,s=1,sa=0,sb=0; ptx::SM100_MMA_MXF4_SS::fma(a,b,c,s,d,sa,sb);'
emit f16bf16_ws      'uint64_t a=0,b=0,d=0; uint32_t c=0,s=1; ptx::SM100_MMA_F16BF16_WS_SS::fma(a,b,c,s,d);'
emit tmem_load_4     'uint32_t v[4]; ptx::tmem_load_32dp32b<4>(0, v);'
emit umma_arrive     'auto* bar = reinterpret_cast<cutlass::arch::ClusterTransactionBarrier*>(0x100); ptx::umma_arrive_no_elect(*bar);'
emit fences          'ptx::tcgen05_before_thread_sync(); ptx::tcgen05_after_thread_sync();'

for op in f16bf16_ss f16bf16_2x1 mxf8f6f4_ss mxf4_ss f16bf16_ws tmem_load_4 umma_arrive fences; do
  s90=FAIL; s100=FAIL
  nvcc -std=c++17 -w -cubin -arch=sm_90a  $INC $TMP/$op.cu -o $TMP/$op.sm90.cubin  >/dev/null 2>$TMP/$op.sm90.log && s90=OK || s90=REJECT
  nvcc -std=c++17 -w -cubin -arch=sm_100a $INC $TMP/$op.cu -o $TMP/$op.sm100.cubin >/dev/null 2>$TMP/$op.sm100.log && s100=OK || s100=REJECT
  printf '  %-14s sm_90a=%-6s sm_100a=%-6s' "$op" "$s90" "$s100"
  if [ "$s90" = REJECT ]; then printf '  | sm90 err: %s' "$(grep -m1 -oE "Feature '[^']*' not supported|Instruction '[^']*' not supported on [^ ]*" $TMP/$op.sm90.log | head -1)"; fi
  echo
done
echo "  (逐条日志见 $TMP/*.log)"

echo
echo "== 编译主实验（sm_90a，真机可运行部分）=="
TORCH=/usr/local/lib/python3.12/dist-packages/torch
PYSITE=/usr/local/lib/python3.12/dist-packages
nvcc -std=c++17 -O2 -w -arch=sm_90a -x cu \
  $INC -I/usr/local/cuda/targets/x86_64-linux/include/cccl \
  -I/usr/local/cuda/include \
  "$SRC" -o /tmp/tcgen05_umma -lcuda -lcudart
echo "build OK"

echo
echo "== 运行 =="
/tmp/tcgen05_umma
EOS
)

"$ROOT/scripts/lab.sh" exec "$CMD"
