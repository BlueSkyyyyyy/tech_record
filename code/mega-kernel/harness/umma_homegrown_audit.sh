#!/usr/bin/env bash
# P160-1 复现：DeepGEMM 自建 PTX UMMA 封装层逐条对照 + 两个 helper + MPK 手写 tcgen05_mma/commit。
#  - host 探针 src/umma_homegrown_audit.cu：逐条对照 9 个 homegrown struct 与 CUTLASS 对应 atom
#    （opcode / operands / elect / disable-output-lane 掩码 / strip_mask 后逐字相等），
#    并对照 tmem_load/umma_arrive helper 与 MPK 手写 tcgen05_mma/commit。
#  - grep audit：DeepGEMM homegrown 9 struct 的调用者、elect_one_sync 守卫分布、
#    MPK 侧 tcgen05_mma/commit 消费者与私有 ptx 副本。
# 纯 host 侧文本审计，不 launch kernel（SM100 本体在目标卡 H100/sm90 不可执行）。
set -u
DG=${DG:-${HOME}/github/DeepGEMM}
MPK=${MPK:-/ssd/mirage_mpk_build}
DGHOME="$DG/deep_gemm/include/deep_gemm/ptx/tcgen05.cuh"
DGCUT="$MPK/deps/cutlass/include/cute/arch/mma_sm100_umma.hpp"
MPKPTX="$MPK/include/mirage/persistent_kernel/tasks/blackwell/sm100_ptx.cuh"
COPY="$MPK/deps/cutlass/include/cute/arch/copy_sm100.hpp"
BARRIER="$MPK/deps/cutlass/include/cutlass/arch/barrier.h"
MPKUTILS="$MPK/include/mirage/persistent_kernel/tasks/blackwell/sm100_utils.cuh"
MPKFP8="$MPK/include/mirage/persistent_kernel/tasks/blackwell/linear_fp8_sm100_sm100_utils.cuh"
MPKDIR="$MPK/include/mirage/persistent_kernel/tasks/blackwell"
SRC=${HOME}/proj/tech_record/code/mega-kernel/src/umma_homegrown_audit.cu
BIN=/tmp/umma_homegrown_audit

g++ -x c++ -std=c++17 -O2 "$SRC" -o "$BIN" || exit 1
echo "\$ $BIN <dghome> <cutlass_mma> <mpk_ptx> <cutlass_copy> <cutlass_barrier> <mpk_utils> <mpk_fp8_utils>"
"$BIN" "$DGHOME" "$DGCUT" "$MPKPTX" "$COPY" "$BARRIER" "$MPKUTILS" "$MPKFP8"
echo

echo "=== audit: DeepGEMM homegrown 9 struct 调用者 (excl. def) ==="
for s in SM100_MMA_F16BF16_SS SM100_MMA_F16BF16_2x1SM_SS \
         SM100_MMA_MXF8F6F4_SS SM100_MMA_MXF8F6F4_2x1SM_SS \
         SM100_MMA_F8F6F4_SS SM100_MMA_F8F6F4_2x1SM_SS \
         SM100_MMA_MXF4_SS SM100_MMA_MXF4_2x1SM_SS \
         SM100_MMA_F16BF16_WS_SS; do
  n=$(grep -rn "ptx::$s\|ptx::$s::fma" "$DG/deep_gemm/include/deep_gemm" "$DG/csrc" 2>/dev/null \
        | grep -v 'ptx/tcgen05.cuh' | wc -l)
  printf '  %-32s %s\n' "$s" "$n"
done
printf '  files #including ptx/tcgen05.cuh: %s\n' \
  "$(grep -rl 'ptx/tcgen05.cuh' "$DG/deep_gemm/include/deep_gemm" 2>/dev/null | wc -l)"
printf '  elect_one_sync call sites under impls/ (homegrown layer has NO internal elect): %s\n' \
  "$(grep -rn 'elect_one_sync' "$DG/deep_gemm/include/deep_gemm/impls" 2>/dev/null | wc -l)"
printf '  tcgen05.fence::* under whole DeepGEMM tree: %s\n' \
  "$(grep -rn 'tcgen05.fence' "$DG/deep_gemm/include/cutlass" "$DG/deep_gemm/include/cute" 2>/dev/null | wc -l)"
echo

echo "=== audit: MPK 手写 sm100_ptx::tcgen05_mma/commit 消费者 (f9eb70c) ==="
printf '  [tcgen05_mma/commit via shared sm100_ptx] (excl. sm100_ptx.cuh):\n'
grep -rn "\btcgen05_mma(\|\btcgen05_commit(" "$MPKDIR" 2>/dev/null \
  | grep -v "sm100_ptx.cuh" | sed 's/^/    /'
printf '  [private namespace ptx copies of tcgen05_mma]:\n'
grep -rln "namespace ptx" "$MPKDIR"/mla_mtp_decode_tp*_sm100.cuh 2>/dev/null | xargs -r -n1 basename | sed 's/^/    /'
printf '  [定义 tcgen05.fence 的 MPK 文件]:\n'
hits=$(grep -l "tcgen05.fence" "$MPKDIR"/*.cuh 2>/dev/null | xargs -r -n1 basename)
if [ -n "$hits" ]; then echo "$hits" | sed 's/^/    /'; else echo "    (none)"; fi
