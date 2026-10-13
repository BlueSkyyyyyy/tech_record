#!/usr/bin/env bash
# P149-1 复现：CUTLASS SM100 稀疏 UMMA 的 *A 操作数* 通路（docs/63）——
#   sparse_elem<2,T> / sparse_ptr / 稀疏 SMEM 布局 SpAtom / sparse_smem_desc /
#   指令描述符 sparse_flag_。补齐 docs/62（P148-1）只走了的 metadata E 侧。
# cute Layout 与 InstrDescriptor 均为 CUTE_HOST_DEVICE，故整套代数在 host 上可
# 实例化/求值，无需 SM100 硬件；用 MPK 快照钉死的 deps/cutlass（f3fde583）。
# 末尾附 consumer audit（grep）：CUTLASS 有稀疏 builder、MPK 零引用。
set -u
DEV=${DEV:-3}
CUTLASS=${CUTLASS:-/ssd/mirage_mpk_build/deps/cutlass}
MPK=${MPK:-/ssd/mirage_mpk_build}
SRC=~/proj/tech_record/code/mega-kernel/src/sparse_mma_a.cu
BIN=/tmp/sparse_mma_a
nvcc -std=c++17 -O2 -arch=sm_90a -I"$CUTLASS/include" "$SRC" -o "$BIN" || exit 1
echo "\$ CUDA_VISIBLE_DEVICES=$DEV $BIN"
CUDA_VISIBLE_DEVICES="$DEV" "$BIN"

echo
echo "=== consumer audit (grep, CUTLASS $CUTLASS) ==="
printf 'sparse MMA_Traits FrgTypeA = sparse_smem_desc : %s\n' \
  "$(grep -rc 'FrgTypeA *= *UMMA::sparse_smem_desc' "$CUTLASS/include/cute/atom/mma_traits_sm100.hpp")"
printf 'sparse MMA_Traits ValTypeA = sparse_elem<2,...> : %s\n' \
  "$(grep -rc 'ValTypeA *= *sparse_elem<2' "$CUTLASS/include/cute/atom/mma_traits_sm100.hpp")"
printf 'sparse K = 512/sizeof_bits<A> sites           : %s\n' \
  "$(grep -rc '512 / cute::sizeof_bits' "$CUTLASS/include/cute/atom/mma_traits_sm100.hpp")"
printf 'SM100 sparse collective builders (CUTLASS)    : %s\n' \
  "$(ls "$CUTLASS/include/cutlass/gemm/collective/builders/" | grep -c sparse)"
echo "  -> $(ls "$CUTLASS/include/cutlass/gemm/collective/builders/" | grep sparse | tr '\n' ' ')"
printf 'MPK (f9eb70c) refs sparse_smem_desc/SpAtom/  : %s\n' \
  "$(grep -rn 'sparse_smem_desc\|sparse_elem\|SpAtom\|make_sparse_ptr' "$MPK/include" "$MPK/src" 2>/dev/null | wc -l)"
printf 'MPK (f9eb70c) refs sparse anywhere (runtime)  : %s\n' \
  "$(grep -rln 'sparse' "$MPK/include/mirage/persistent_kernel" 2>/dev/null | wc -l)"
