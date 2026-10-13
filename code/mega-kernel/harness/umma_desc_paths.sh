#!/usr/bin/env bash
# P159-1 复现（docs/73）：UMMA SMEM 描述符构造的「多路并置」——
#   [母本] DeepGEMM deep_gemm/mma/sm100.cuh
#   [全量] MPK linear_fp8_sm100_sm100_utils.cuh
#   [子集] MPK sm100_utils.cuh
#   [手写] MPK sm100_ptx.cuh::make_desc
# host 侧逐字重放四者的构造算术, 打印真实 64 位描述符并逐位对比。
# 补 docs/72 §7 明留 + ROADMAP「下一步候选」第一条（描述符构造并置, 未与 DeepGEMM 自建层对走）。
# 无需 SM100 硬件（目标卡 H100/sm90 也没有）。
# 快照: mirage f9eb70c / CUTLASS f3fde583 / DeepGEMM 057ca59。
set -u
DEV=${DEV:-3}
MPK=${MPK:-/ssd/mirage_mpk_build}
CUTLASS=${CUTLASS:-$MPK/deps/cutlass}
DEEPGEMM=${DEEPGEMM:-~/github/DeepGEMM}
SRC=~/proj/tech_record/code/mega-kernel/src/umma_desc_paths.cu
BIN=/tmp/umma_desc_paths
nvcc -std=c++17 -O2 -arch=sm_90a -I"$CUTLASS/include" "$SRC" -o "$BIN" || exit 1
echo "\$ CUDA_VISIBLE_DEVICES=$DEV $BIN"
CUDA_VISIBLE_DEVICES="$DEV" "$BIN"

BW="$MPK/include/mirage/persistent_kernel/tasks/blackwell"
DG="$DEEPGEMM/deep_gemm/include/deep_gemm"

echo
echo "=== 四路坐标 ==="
printf '  %-46s %s\n' "DeepGEMM 自建 (母本)" "$DG/mma/sm100.cuh"
printf '  %-46s %s\n' "MPK 全量副本"        "$BW/linear_fp8_sm100_sm100_utils.cuh"
printf '  %-46s %s\n' "MPK K-major 子集"    "$BW/sm100_utils.cuh"
printf '  %-46s %s\n' "MPK 手写 make_desc"  "$BW/sm100_ptx.cuh"

echo
echo "=== helper 存在性 (首个出现行号) ==="
for f in "$DG/mma/sm100.cuh" "$BW/linear_fp8_sm100_sm100_utils.cuh" "$BW/sm100_utils.cuh"; do
  printf '  %-40s lines=%s\n' "$(basename "$f")" "$(wc -l < "$f")"
  for sym in make_smem_desc make_sf_desc replace_smem_desc_addr to_umma_layout_type \
             get_atom_base get_umma_desc_stride_k advance_umma_desc_lo make_umma_desc \
             make_runtime_instr_desc_with_sf_id update_instr_desc_with_umma_n; do
    ln=$(grep -n "\b$sym\b" "$f" | head -1 | cut -d: -f1)
    printf '      %-36s %s\n' "$sym" "${ln:--}"
  done
done

echo
echo "=== 「子集 vs 全量」分支差异 (K-major-only static_assert) ==="
grep -n "Current Mirage helper only supports K-major" "$BW/sm100_utils.cuh" | sed 's/^/    /'
grep -n "SWIZZLE_128B;" "$BW/sm100_utils.cuh" | sed 's/^/    /'

echo
echo "=== DeepGEMM 母本里的 pack-factor 支持 (MPK bootstrap removed) ==="
grep -n "kPackFactor\|get_smem_pack_factor" "$DG/mma/sm100.cuh" | sed 's/^/    /'
grep -n "get_smem_pack_factor" "$DG/common/packing.cuh" | sed 's/^/    /'

echo
echo "=== DeepGEMM 母本消费者 (谁 include mma/sm100.cuh) ==="
grep -rln 'include <deep_gemm/mma/sm100.cuh>\|include "deep_gemm/mma/sm100.cuh"' "$DEEPGEMM/deep_gemm/include" | xargs -r -n1 basename | sort | sed 's/^/    /'
