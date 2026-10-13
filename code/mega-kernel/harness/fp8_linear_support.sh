#!/usr/bin/env bash
# P154-1 复现（docs/68）：MPK SM100 block-scaled FP8 linear 的「支撑层」——
#   linear_fp8_sm100_utils.cuh      (143 行)
#   linear_fp8_sm100_sm100_utils.cuh(232 行)
#   linear_fp8_sm100_tma_utils.cuh  (118 行)
# 逐行打开，并把其中「纯算术」（SmemDescriptor / make_umma_desc /
# InstrDescriptorBlockScaled / get_num_aligned_tmem_cols）在 host 上逐字重放、打印
# 真实描述符位模式。补两份明留的尾巴：
#   * docs/64 §12: "`_tma_utils` / `_sm100_utils` 只借用了行号，未逐行"
#   * docs/57 §8 : "`make_smem_desc` 的 version_/lbo_mode_ 语义未逐位展开"
# 无需 SM100 硬件（目标卡 H100/sm90 也没有）。快照 mirage f9eb70c / CUTLASS f3fde583。
set -u
DEV=${DEV:-3}
MPK=${MPK:-/ssd/mirage_mpk_build}
CUTLASS=${CUTLASS:-$MPK/deps/cutlass}
SRC=~/proj/tech_record/code/mega-kernel/src/fp8_linear_support.cu
BIN=/tmp/fp8_linear_support
nvcc -std=c++17 -O2 -arch=sm_90a -I"$CUTLASS/include" "$SRC" -o "$BIN" || exit 1
echo "\$ CUDA_VISIBLE_DEVICES=$DEV $BIN"
CUDA_VISIBLE_DEVICES="$DEV" "$BIN"

BW="$MPK/include/mirage/persistent_kernel/tasks/blackwell"
TMA="$BW/linear_fp8_sm100_tma_utils.cuh"
SM100U="$BW/linear_fp8_sm100_sm100_utils.cuh"
U="$BW/linear_fp8_sm100_utils.cuh"

echo
echo "=== 支撑层 include 链（MPK $MPK @ f9eb70c）==="
printf '  %-42s %5s lines\n' "linear_fp8_sm100_utils.cuh"       "$(wc -l < "$U")"
printf '  %-42s %5s lines\n' "linear_fp8_sm100_tma_utils.cuh"   "$(wc -l < "$TMA")"
printf '  %-42s %5s lines\n' "linear_fp8_sm100_sm100_utils.cuh" "$(wc -l < "$SM100U")"
echo "  谁 include 这三支（blackwell/ 内）:"
grep -rn "include \"linear_fp8_sm100_\(tma_utils\|sm100_utils\|utils\).cuh\"" "$BW" | sed 's/^/    /'

echo
echo "=== helper -> 调用方（blackwell/ 内，含定义文件本身）==="
for sym in make_smem_desc make_sf_desc replace_smem_desc_addr to_umma_layout_type \
           make_umma_desc make_runtime_instr_desc_with_sf_id get_num_aligned_tmem_cols \
           tcgen05_before_thread_sync SM100_MMA_MXF8F6F4_SS SM100_MMA_MXF8F6F4_2x1SM_SS \
           tma_copy tensor_map_release_cta tensor_map_acquire_cta \
           tensor_map_replace_global_addr_in_smem; do
  files=$(grep -rl "\b$sym\b" "$BW" 2>/dev/null | xargs -r -n1 basename | tr '\n' ' ')
  printf '  %-40s %s\n' "$sym" "$files"
done

echo
echo "=== 「同名 helper 定义两遍」普查：linear_fp8_sm100_sm100_utils.cuh  vs  sm100_utils.cuh ==="
for sym in make_smem_desc make_sf_desc make_umma_desc make_runtime_instr_desc_with_sf_id \
           advance_umma_desc_lo make_runtime_instr_desc_with_sf_id SM100_MMA_MXF8F6F4_SS; do
  a=$(grep -c "\b$sym\b" "$SM100U")
  b=$(grep -c "\b$sym\b" "$BW/sm100_utils.cuh")
  printf '  %-40s linear_fp8=%s  sm100_utils=%s\n' "$sym" "$a" "$b"
done

echo
echo "=== 「有实现、零调用者」普查（全 MPK 树，定义处之外）==="
for sym in tensor_map_release_cta tensor_map_acquire_cta tensor_map_replace_global_addr_in_smem; do
  n=$(grep -rn "\b$sym\b" "$MPK/include" "$MPK/src" 2>/dev/null \
        | grep -v "linear_fp8_sm100_tma_utils.cuh" | wc -l)
  printf '  %-40s callers(outside def) = %s\n' "$sym" "$n"
done

echo
echo "=== 现有消费者（body A / body B / 老 sm100_utils 支）对照 ==="
printf '  %-42s tma_copy=%s PatternVisitor=%s\n' "linear_fp8_sm100.cuh (body A)" \
  "$(grep -c 'tma_copy' "$BW/linear_fp8_sm100.cuh")" "$(grep -c 'PatternVisitor' "$BW/linear_fp8_sm100.cuh")"
printf '  %-42s tma_copy=%s PatternVisitor=%s\n' "linear_fp8_1d2d_sm100.cuh (body B)" \
  "$(grep -c 'tma_copy' "$BW/linear_fp8_1d2d_sm100.cuh")" "$(grep -c 'PatternVisitor' "$BW/linear_fp8_1d2d_sm100.cuh")"
printf '  %-42s tma_copy=%s PatternVisitor=%s\n' "fp8_group_gemm_sm100.cuh (MoE)" \
  "$(grep -c 'tma_copy' "$BW/fp8_group_gemm_sm100.cuh")" "$(grep -c 'PatternVisitor' "$BW/fp8_group_gemm_sm100.cuh")"
