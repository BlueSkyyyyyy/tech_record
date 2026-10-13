#!/usr/bin/env bash
# P148-1 复现：CUTLASS SM100 UMMA TMEM fragment trait 家族的「其余两面」——
# 稀疏 MMA 元数据 E（tmem_e_frg / tmem_e_frg_ws）与 weight-stationary C/D（tmem_frg_ws）
# 的 host 侧结构探测（目标卡 H100 / sm90a）。
# cute Layout 是 CUTE_HOST_DEVICE，故整套布局代数可在 host 上实例化/求值，无需 SM100 硬件；
# 用 MPK 快照钉死的 deps/cutlass（f3fde583）的真实头文件；末尾附 consumer audit（grep）。
set -u
DEV=${DEV:-3}
CUTLASS=${CUTLASS:-/ssd/mirage_mpk_build/deps/cutlass}
MPK=${MPK:-/ssd/mirage_mpk_build}
SRC=~/proj/tech_record/code/mega-kernel/src/tmem_frg_traits.cu
BIN=/tmp/tmem_frg_traits
nvcc -std=c++17 -O2 -arch=sm_90a -I"$CUTLASS/include" "$SRC" -o "$BIN" || exit 1
echo "\$ CUDA_VISIBLE_DEVICES=$DEV $BIN"
CUDA_VISIBLE_DEVICES="$DEV" "$BIN"

echo
echo "=== consumer audit (grep, CUTLASS $CUTLASS) ==="
printf 'FrgTypeE = UMMA::tmem_e_frg<...>    : %s\n' \
  "$(grep -rn 'FrgTypeE *= *UMMA::tmem_e_frg<' "$CUTLASS/include" | wc -l)"
printf 'FrgTypeE = UMMA::tmem_e_frg_ws<...> : %s\n' \
  "$(grep -rn 'FrgTypeE *= *UMMA::tmem_e_frg_ws' "$CUTLASS/include" | wc -l)"
printf 'FrgTypeC/D = UMMA::tmem_frg_ws<...> : %s\n' \
  "$(grep -rn 'FrgType[CD] *= *UMMA::tmem_frg_ws' "$CUTLASS/include" | wc -l)"
printf 'any SM100_MMA_*_WS* atom struct     : %s\n' \
  "$(grep -rn 'struct SM100_MMA.*WS\|SM100_MMA.*WeightStationary' "$CUTLASS/include" | wc -l)"
printf 'MPK (f9eb70c) refs tmem_e_frg/frg_ws: %s\n' \
  "$(grep -rn 'tmem_e_frg\|tmem_frg_ws' "$MPK/include" "$MPK/src" 2>/dev/null | wc -l)"
echo "sparse MMA_Traits that consume tmem_e_frg as FrgTypeE:"
grep -rn 'FrgTypeE *= *UMMA::tmem_e_frg<' "$CUTLASS/include/cute/atom/mma_traits_sm100.hpp" \
  | sed 's/^/  /'
