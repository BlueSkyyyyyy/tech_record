#!/usr/bin/env bash
# P157-1 复现：WS（weight-stationary）UMMA 指令落地现状对照。
#  - host 侧结构探测（目标卡 H100 / sm90a）：max_shift_ 描述符位、WS C/D 与
#    WS 稀疏 E 的 TMEM fragment 形状（用 DeepGEMM vendored CUTLASS，与 MPK 的
#    deps/cutlass 同为 f3fde583）。
#  - audit（grep）：三方 CUTLASS 头 md5、`.mma.ws` 出现处、DeepGEMM 自建 PTX
#    UMMA 封装层的 9 个 struct 的调用者数、MaxShift 引用、WS trait 消费者数。
set -u
DEV=${DEV:-3}
DG=${DG:-${HOME}/github/DeepGEMM}
MPK=${MPK:-/ssd/mirage_mpk_build}
DGCUTLASS="$DG/deep_gemm/include"
TPCUTLASS="$DG/third-party/cutlass/include"
MPKCUTLASS="$MPK/deps/cutlass/include"
SRC=${HOME}/proj/tech_record/code/mega-kernel/src/ws_instruction_probe.cu
BIN=/tmp/ws_instruction_probe
nvcc -std=c++17 -O2 -arch=sm_90a -I"$DGCUTLASS" "$SRC" -o "$BIN" || exit 1
echo "\$ CUDA_VISIBLE_DEVICES=$DEV $BIN"
CUDA_VISIBLE_DEVICES="$DEV" "$BIN"

echo
echo "=== audit: three CUTLASS checkouts are the same commit f3fde583 (md5) ==="
for f in include/cute/atom/mma_traits_sm100.hpp \
         include/cute/arch/mma_sm100_desc.hpp \
         include/cute/arch/mma_sm100_umma.hpp; do
  printf '%s\n' "$f"
  printf '  deep_gemm/include  : %s\n' "$(md5sum "$DG/deep_gemm/$f" 2>/dev/null | cut -d' ' -f1)"
  printf '  third-party/cutlass: %s\n' "$(md5sum "$DG/third-party/cutlass/$f" 2>/dev/null | cut -d' ' -f1)"
  printf '  MPK deps/cutlass   : %s\n' "$(md5sum "$MPK/deps/cutlass/$f" 2>/dev/null | cut -d' ' -f1)"
done
printf 'deep_gemm/include/cute/... WS atoms (struct SM100_MMA.*_WS): %s\n' \
  "$(grep -rn 'struct SM100_MMA.*_WS\|WeightStationary' "$DGCUTLASS/cute/arch" 2>/dev/null | wc -l)"
printf 'MPK deps/cutlass     WS atoms (struct SM100_MMA.*_WS): %s\n' \
  "$(grep -rn 'struct SM100_MMA.*_WS\|WeightStationary' "$MPKCUTLASS/cute/arch" 2>/dev/null | wc -l)"

echo
echo "=== audit: the single tcgen05.mma.ws occurrence in DeepGEMM ==="
grep -rn 'mma\.ws' "$DG/deep_gemm/include" "$DG/csrc" 2>/dev/null | sed 's/^/  /'
printf '  MPK (f9eb70c) .mma.ws occurrences: %s\n' \
  "$(grep -rn 'mma\.ws' "$MPK/include" "$MPK/src" "$MPK/python" 2>/dev/null | wc -l)"

echo
echo "=== audit: DeepGEMM homegrown PTX UMMA wrapper set callers (excl. def) ==="
for s in SM100_MMA_F16BF16_SS SM100_MMA_F16BF16_2x1SM_SS \
         SM100_MMA_MXF8F6F4_SS SM100_MMA_MXF8F6F4_2x1SM_SS \
         SM100_MMA_F8F6F4_SS SM100_MMA_F8F6F4_2x1SM_SS \
         SM100_MMA_MXF4_SS SM100_MMA_MXF4_2x1SM_SS \
         SM100_MMA_F16BF16_WS_SS; do
  n=$(grep -rn "\b$s\b" "$DG/deep_gemm/include/deep_gemm" "$DG/csrc" 2>/dev/null \
        | grep -v 'ptx/tcgen05.cuh' | wc -l)
  printf '  %-28s : %s refs (excl. def)\n' "$s" "$n"
done

echo
echo "=== audit: MaxShift / max_shift in DeepGEMM own code and MPK own code ==="
printf '  DeepGEMM own code MaxShift/max_shift refs : %s\n' \
  "$(grep -rn 'MaxShift\|max_shift' "$DG/deep_gemm/include/deep_gemm" "$DG/csrc" 2>/dev/null | wc -l)"
printf '  MPK own code     MaxShift/max_shift refs : %s\n' \
  "$(grep -rn 'MaxShift\|max_shift' "$MPK/include" "$MPK/src" "$MPK/python" 2>/dev/null | wc -l)"

echo
echo "=== audit: WS trait consumers (FrgType* = *_ws) in both CUTLASS copies ==="
for C in "$DGCUTLASS" "$MPKCUTLASS"; do
  echo "  [$C]"
  printf '    FrgTypeC/D = UMMA::tmem_frg_ws : %s\n' \
    "$(grep -rn 'FrgType[CD] *= *UMMA::tmem_frg_ws' "$C" 2>/dev/null | wc -l)"
  printf '    FrgTypeE   = UMMA::tmem_e_frg_ws : %s\n' \
    "$(grep -rn 'FrgTypeE *= *UMMA::tmem_e_frg_ws' "$C" 2>/dev/null | wc -l)"
done
