#!/usr/bin/env bash
# P158-1 复现：UMMA 原语多实现并置审计（纯 host 侧，不 launch kernel）。
#  - host 探针 src/umma_paths_audit.cu：抽各层 tcgen05.mma 指令、出 capability 矩阵、
#    核对四份 mxf8f6f4.block_scale wrapper 的逐字相等性。
#  - grep audit：DeepGEMM 自建层的 9 struct 调用者、elect_one_sync 守卫、MPK 侧各层消费者。
set -u
DG=${DG:-~/github/DeepGEMM}
MPK=${MPK:-/ssd/mirage_mpk_build}
DGHOME="$DG/deep_gemm/include/deep_gemm/ptx/tcgen05.cuh"
DGCUT="$DG/deep_gemm/include/cute/arch/mma_sm100_umma.hpp"
MPKDIR="$MPK/include/mirage/persistent_kernel/tasks/blackwell"
MPKCUT="$MPK/deps/cutlass/include/cute/arch/mma_sm100_umma.hpp"
SRC=~/proj/tech_record/code/mega-kernel/src/umma_paths_audit.cu
BIN=/tmp/umma_paths_audit

g++ -x c++ -std=c++17 -O2 "$SRC" -o "$BIN" || exit 1
echo "\$ $BIN <dghome> <dg_cutlass_mma> <mpk_blackwell_dir> <mpk_cutlass_mma>"
"$BIN" "$DGHOME" "$DGCUT" "$MPKDIR" "$MPKCUT"
echo

echo "=== audit: DeepGEMM homegrown PTX UMMA wrapper: callers per struct (excl. def) ==="
for s in SM100_MMA_F16BF16_SS SM100_MMA_F16BF16_2x1SM_SS \
         SM100_MMA_MXF8F6F4_SS SM100_MMA_MXF8F6F4_2x1SM_SS \
         SM100_MMA_F8F6F4_SS SM100_MMA_F8F6F4_2x1SM_SS \
         SM100_MMA_MXF4_SS SM100_MMA_MXF4_2x1SM_SS \
         SM100_MMA_F16BF16_WS_SS; do
  n=$(grep -rn "\b$s\b" "$DG/deep_gemm/include/deep_gemm" "$DG/csrc" 2>/dev/null \
        | grep -v 'ptx/tcgen05.cuh' | wc -l)
  printf '  %-32s %s\n' "$s" "$n"
done
printf '  files #including ptx/tcgen05.cuh: %s\n' \
  "$(grep -rl 'ptx/tcgen05.cuh' "$DG/deep_gemm/include/deep_gemm" 2>/dev/null | wc -l)"
printf '  call sites guarding with cute::elect_one_sync(): %s (homegrown layer has NO internal elect)\n' \
  "$(grep -rn 'elect_one_sync' "$DG/deep_gemm/include/deep_gemm/impls" 2>/dev/null | wc -l)"
echo

echo "=== audit: MPK UMMA paths consumers (f9eb70c) ==="
printf '  [kernel::sm100::SM100_MMA_MXF8F6F4_SS] consumers (excl. def):\n'
grep -rn "sm100::SM100_MMA_MXF8F6F4_SS" "$MPK/include" "$MPK/src" 2>/dev/null \
  | grep -v "sm100_utils.cuh" | sed 's/^/    /'
printf '  [mirage::blackwell::linear_fp8_sm100::sm100::SM100_MMA_*] consumers (excl. def):\n'
grep -rn "linear_fp8_sm100::sm100::SM100_MMA" "$MPK/include" "$MPK/src" 2>/dev/null \
  | grep -v "linear_fp8_sm100_sm100_utils.cuh" | sed 's/^/    /'
printf '  [cute::SM100_MMA_F16BF16_SS] consumers:\n'
grep -rn "cute::SM100_MMA_F16BF16_SS" "$MPKDIR" 2>/dev/null | sed 's/^/    /'
printf '  [sm100_ptx::tcgen05_mma] consumer files:\n'
grep -rl "sm100_ptx::tcgen05_mma\|tcgen05_mma(out_taddr\|tcgen05_mma(out_taddr" "$MPKDIR" 2>/dev/null \
  | xargs -r -n1 basename | sed 's/^/    /'
echo

echo "=== audit: MPK private ptx namespace copies of tcgen05_mma (f9eb70c) ==="
grep -rln "namespace ptx" "$MPKDIR"/mla_*_sm100.cuh 2>/dev/null | sed 's/^/    /'
