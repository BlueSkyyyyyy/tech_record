#!/usr/bin/env bash
# P156-1 复现：结构稀疏（2:4）元数据 E 的 *内容*——位编码 + A 物理槽置换（docs/70）。
# 补齐 docs/62（P148-1，只走 E 的 TMEM trait 形状）与 docs/63（P149-1，只走 A 的
# 类型/SMEM 布局）。整套算术（SparseConfig 常量、CUTE 布局、compressor switch/loop）
# 都是 CUTE_HOST_DEVICE，故在 host 上可判定，无需 SM90/SM100 张量核。
# 用 MPK 快照钉死的 deps/cutlass（f3fde583）。末尾附 consumer audit（grep）。
set -u
DEV=${DEV:-3}
CUTLASS=${CUTLASS:-/ssd/mirage_mpk_build/deps/cutlass}
MPK=${MPK:-/ssd/mirage_mpk_build}
SRC=${HOME}/proj/tech_record/code/mega-kernel/src/sparse_e_metadata.cu
BIN=/tmp/sparse_e_metadata
nvcc -std=c++17 -O2 -arch=sm_90a -I"$CUTLASS/include" "$SRC" -o "$BIN" || exit 1
echo "\$ CUDA_VISIBLE_DEVICES=$DEV $BIN"
CUDA_VISIBLE_DEVICES="$DEV" "$BIN"

echo
echo "=== consumer audit (grep, CUTLASS $CUTLASS) ==="
printf 'SM100 SparseConfig (Sm1xxGemmSparseConfig) defs   : %s\n' \
  "$(grep -rc 'struct Sm1xxGemmSparseConfig' "$CUTLASS/include/cutlass/gemm/collective/builders/sm1xx_sparse_config.inl")"
printf 'SM90  SparseConfig (Sm90GemmSparseConfig) defs    : %s\n' \
  "$(grep -rc 'struct Sm90GemmSparseConfig' "$CUTLASS/include/cutlass/gemm/collective/builders/sm90_sparse_config.inl")"
printf 'E encoder structs (MetadataOneChunk*to*)         : %s\n' \
  "$(grep -rc 'struct MetadataOneChunk' "$CUTLASS/include/cutlass/transform/kernel/sm90_sparse_gemm_compressor.hpp")"
printf 'StructuredSparseCompressor selector arms          : %s\n' \
  "$(grep -rc 'using Compressor = SM90StructuredSparseCompressor' "$CUTLASS/include/cutlass/transform/kernel/sparse_gemm_compressor.hpp")"
printf 'MPK (f9eb70c) refs Sm1xx/Sm90GemmSparseConfig    : %s\n' \
  "$(grep -rn 'GemmSparseConfig\|StructuredSparseCompressor' "$MPK/include" "$MPK/src" 2>/dev/null | wc -l)"
printf 'DeepGEMM (057ca59) refs GemmSparseConfig/compressor: %s\n' \
  "$(grep -rln 'GemmSparseConfig\|StructuredSparseCompressor' ${HOME}/github/DeepGEMM/csrc 2>/dev/null | wc -l)"
printf 'DeepGEMM "sparse" files (block-sparse, NOT 2:4)  : %s\n' \
  "$(grep -rln -i 'sparse' ${HOME}/github/DeepGEMM/csrc 2>/dev/null | wc -l)"
