#!/usr/bin/env bash
# P7-41: ncu the CUTLASS-`PipelineTmaAsync` B1 vs the hand-written-mbarrier B1 (same operands,
# same stores/fences), to see whether the residual §24.5 stall (`wait` = mma fixed latency,
# `short_sb` = smem operand fetch) moves at all when the pipeline protocol is swapped.
#
# Pre-build the .so outside ncu (nvcc under ncu hangs), then profile with --no-build.
#
# Usage (inside kernel_lab):
#   bash /ssd/.../harness/probe_b1_cutlass_ncu.sh \
#       > /ssd/.../harness/probe_b1_cutlass_ncu.out.txt 2>&1
set -uo pipefail

DIR=~/proj/tech_record/code/flash-kda
DRIVER=$DIR/harness/ncu_b1_cutlass_driver.py
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-1}
cd "$DIR"

SECTIONS="--section SpeedOfLight --section Occupancy --section SchedulerStats \
--section WarpStateStats --section LaunchStats --section MemoryWorkloadAnalysis \
--section ComputeWorkloadAnalysis"

METRICS="sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active,\
smsp__average_warps_issue_stalled_barrier_per_issue_active.ratio,\
smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio,\
smsp__average_warps_issue_stalled_short_scoreboard_per_issue_active.ratio,\
smsp__average_warps_issue_stalled_wait_per_issue_active.ratio,\
smsp__average_warps_issue_stalled_not_selected_per_issue_active.ratio,\
l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum,\
l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum,\
l1tex__throughput.avg.pct_of_peak_sustained_elapsed,\
lts__throughput.avg.pct_of_peak_sustained_elapsed,\
dram__throughput.avg.pct_of_peak_sustained_elapsed,\
sm__throughput.avg.pct_of_peak_sustained_elapsed"

tag() { sed -e '/^==PROF==/d' -e '/^==WARNING==/d'; }

# prebuild all variants once, outside ncu
for v in hnd cut2; do
  python "$DRIVER" --variant "$v" --B 8 --T 2048 --H 8 --bt 64 --ns 2 >/dev/null 2>&1
  python "$DRIVER" --variant "$v" --B 1 --T 512  --H 4 --bt 64 --ns 2 >/dev/null 2>&1
done

run() {  # label variant B T H bt ns
  local label=$1 variant=$2 B=$3 T=$4 H=$5 bt=$6 ns=$7
  echo ""
  echo "################################################################################"
  echo "# ${label}"
  echo "#   variant=${variant} B=${B} T=${T} H=${H} bt=${bt} ns=${ns} kernel=b1_tma_kernel"
  echo "################################################################################"
  ncu -k "regex:b1_tma_kernel" -c 1 --launch-skip 3 $SECTIONS --metrics "$METRICS" \
      python "$DRIVER" --variant "$variant" --B "$B" --T "$T" --H "$H" --bt "$bt" \
      --ns "$ns" --no-build 2>&1 | tag | grep -vE '^DONE'
}

echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "ncu: $(ncu --version | grep -i 'version' | head -1)"

run "HND   (hand mbarrier)  B8 T2048 H8 BT=64 NS=2" hnd  8 2048 8 64 2
run "CUT2  (CUTLASS pipe)   B8 T2048 H8 BT=64 NS=2" cut2 8 2048 8 64 2
run "HND   (hand mbarrier)  B1 T512  H4 BT=64 NS=2" hnd  1 512  4 64 2
run "CUT2  (CUTLASS pipe)   B1 T512  H4 BT=64 NS=2" cut2 1 512  4 64 2

echo ""
echo "DONE"
