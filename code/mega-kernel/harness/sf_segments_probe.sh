#!/usr/bin/env bash
# P155-1 (docs/69) 复现：MPK SM100 block-scaled FP8 的「SF 段」并置走读的 host 侧算术探针。
# 对走 linear_fp8_sm100.cuh (dense) 与 fp8_group_gemm_sm100.cuh (grouped MoE) 的 SF 段，
# 重放 UE8M0 转换 / dense vs MoE 打包 / 行数账 / 4x32 转置双射。
# 无需 SM100 硬件（目标卡 H100/sm90 也没有）。快照 mirage f9eb70c。
set -u
DEV=${DEV:-3}
SRC=${HOME}/proj/tech_record/code/mega-kernel/src/sf_segments_probe.cu
BIN=/tmp/sf_segments_probe
nvcc -std=c++17 -O2 -arch=sm_90a "$SRC" -o "$BIN" || exit 1
echo "\$ CUDA_VISIBLE_DEVICES=$DEV $BIN"
CUDA_VISIBLE_DEVICES="$DEV" "$BIN"

MPK=${MPK:-/ssd/mirage_mpk_build}
BW="$MPK/include/mirage/persistent_kernel/tasks/blackwell"
DENSE="$BW/linear_fp8_sm100.cuh"
MOE="$BW/fp8_group_gemm_sm100.cuh"

echo
echo "=== SF 段关键锚点对照（MPK $MPK @ f9eb70c）==="
printf '  %-34s %5s lines\n' "linear_fp8_sm100.cuh"    "$(wc -l < "$DENSE")"
printf '  %-34s %5s lines\n' "fp8_group_gemm_sm100.cuh" "$(wc -l < "$MOE")"
echo
printf '  %-52s %s\n' "dense kNumS*StagesPerLoad" "$(grep -n 'kNumSFAStagesPerLoad =' "$DENSE" | head -1)"
printf '  %-52s %s\n' "dense shape_sf*_k" "$(grep -n 'shape_sf._k = ceil_div' "$DENSE")"
printf '  %-52s %s\n' "dense sfa_id/sfb_id" "$(grep -n 'uint32_t const sfa_id =' "$DENSE")"
printf '  %-52s %s\n' "MoE 4-bytes-identical 注释" "$(grep -n 'identical replicas' "$MOE")"
printf '  %-52s %s\n' "MoE sfa_id=k_sub" "$(grep -n 'uint32_t const sfa_id = k_sub' "$MOE")"
printf '  %-52s %s\n' "MoE FIX ab_empty" "$(grep -n 'FIX 2026-04-22' "$MOE")"
echo
echo "=== 生产实例化参数（src/kernel/task_register.cc @ 4772-4848）==="
grep -n '128, 128,"); // kGranKA\|32, 16, 128,");  // BLOCK\|base Offset\|MPK_MAX_NUM_WORKERS\|kNumSMs' \
  "$MPK/src/kernel/task_register.cc" | sed -n '1,6p' | sed 's/^/  /'
