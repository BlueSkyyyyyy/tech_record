#!/usr/bin/env bash
# P150-1 复现（docs/64）：MPK SM100 block-scaled FP8 linear 家族的内核内 tile
# 调度器（linear_fp8_sm100_scheduler.cuh）走读 —— 补 docs/32（P119-1）§2.3 /
# §7 只映射了行号区间（"`_scheduler`/`_epilogue_utils`/`_cute_tie` 未逐行走"）
# 的尾巴。调度器方法都是 __device__，但算术是纯组合，可在 host 上逐字重放并
# 校验（双射 / 覆盖）。末尾附 MPK vs DeepGEMM 调度器的差异普查（grep）。
#
# 无需 SM100 硬件（目标卡 H100/sm90 也没有）。快照钉死 f9eb70c。
set -u
DEV=${DEV:-3}
MPK=${MPK:-/ssd/mirage_mpk_build}
DEEPGEMM=${DEEPGEMM:-~/github/DeepGEMM}
SRC=~/proj/tech_record/code/mega-kernel/src/fp8_scheduler_order.cu
BIN=/tmp/fp8_scheduler_order
nvcc -std=c++17 -O2 -arch=sm_90a "$SRC" -o "$BIN" || exit 1
echo "\$ CUDA_VISIBLE_DEVICES=$DEV $BIN"
CUDA_VISIBLE_DEVICES="$DEV" "$BIN"

MPK_SCHED="$MPK/include/mirage/persistent_kernel/tasks/blackwell/linear_fp8_sm100_scheduler.cuh"
MPK_EPI="$MPK/include/mirage/persistent_kernel/tasks/blackwell/linear_fp8_sm100_epilogue_utils.cuh"
DG_SCHED="$DEEPGEMM/deep_gemm/include/deep_gemm/scheduler/gemm.cuh"

echo
echo "=== scheduler census (MPK $MPK @ f9eb70c  vs  DeepGEMM $DEEPGEMM @ $(git -C "$DEEPGEMM" rev-parse --short HEAD 2>/dev/null)) ==="
printf '%-42s MPK=%5s  DeepGEMM=%5s\n' "file lines" \
  "$(wc -l < "$MPK_SCHED")" "$(wc -l < "$DG_SCHED")"
for sym in get_num_1d_blocks_per_group get_swizzled_block_idx get_global_idx get_next_block is_peer_cta_alive is_tma_multicast_valid get_aligned_effective_m_in_block get_next_k_group; do
  printf '%-42s MPK=%5s  DeepGEMM=%5s\n' "$sym" \
    "$(grep -c "$sym" "$MPK_SCHED")" "$(grep -c "$sym" "$DG_SCHED")"
done
echo
echo "MPK get_next_block handles GemmType branches:"
grep -n "GemmType::" "$MPK_SCHED" | sed 's/^/    MPK   /'
echo "DeepGEMM get_next_block GemmType branches:"
grep -n "GemmType::" "$DG_SCHED" | grep -v get_global_idx | sed 's/^/    DG    /'

echo
echo "MPK _scheduler.cuh hard-coded tile offset (must be +0 for the persistent launch):"
grep -n "block_offset\|++current_iter" "$MPK_SCHED" | sed 's/^/    /'
echo "DeepGEMM equivalent (uses blockIdx.x):"
grep -n "current_iter) \* kNumSMs" "$DG_SCHED" | sed 's/^/    /'
echo "task_register HACK comment vs emitted kNumSMs:"
grep -n "MPK_MAX_NUM_WORKERS\|persistent kernel: 1 CTA per task" "$MPK/src/kernel/task_register.cc" | sed 's/^/    /'
echo "octothorpe audit: MPK_MAX_NUM_WORKERS occurrences in whole MPK snapshot:"
grep -rn "MPK_MAX_NUM_WORKERS" "$MPK/include" "$MPK/src" 2>/dev/null | wc -l | sed 's/^/    /'
