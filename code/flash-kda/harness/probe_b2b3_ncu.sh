#!/usr/bin/env bash
# P7-23: ncu bound analysis for the two per-chunk backward GEMM kernels B2 (`wy_dqkg`) and
# B3 (`intra`) at the deployed CHUNK=16 granularity.  B1 was profiled in P7-13/P7-16/P7-19;
# the P7-22 chain split shows B2+B3 = ~66% of backward device time, so their bound is the
# open question.  We read SpeedOfLight / Occupancy / stall reasons / memory traffic.
#
# Usage (inside kernel_lab):
#   bash .../probe_b2b3_ncu.sh > .../probe_b2b3_ncu.out.txt 2>&1

set -uo pipefail

DIR=~/proj/tech_record/code/flash-kda
DRIVER=$DIR/harness/ncu_b2b3_driver.py
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-1}
cd "$DIR"

SECTIONS="--section SpeedOfLight --section Occupancy --section SchedulerStats \
--section WarpStateStats --section LaunchStats --section MemoryWorkloadAnalysis"

METRICS="sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active,\
sm__pipe_fma_cycles_active.avg.pct_of_peak_sustained_active,\
sm__inst_executed_pipe_tensor.sum,\
smsp__average_warps_issue_stalled_barrier_per_issue_active.ratio,\
smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio,\
smsp__average_warps_issue_stalled_short_scoreboard_per_issue_active.ratio,\
smsp__average_warps_issue_stalled_wait_per_issue_active.ratio,\
smsp__average_warps_issue_stalled_mio_throttle_per_issue_active.ratio,\
smsp__average_warps_issue_stalled_not_selected_per_issue_active.ratio,\
smsp__average_warps_issue_stalled_lg_throttle_per_issue_active.ratio,\
smsp__average_warps_issue_stalled_drain_per_issue_active.ratio,\
smsp__average_warps_issue_stalled_imc_miss_per_issue_active.ratio,\
l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum,\
l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum,\
l1tex__throughput.avg.pct_of_peak_sustained_elapsed,\
lts__throughput.avg.pct_of_peak_sustained_elapsed,\
dram__throughput.avg.pct_of_peak_sustained_elapsed,\
dram__bytes_read.sum,dram__bytes_write.sum,lts__t_bytes.sum,\
sm__throughput.avg.pct_of_peak_sustained_elapsed,\
gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed"

tag() { sed -e '/^==PROF==/d' -e '/^==WARNING==/d'; }

run() {  # label which B T H bt nw kernel_regex
  local label=$1 which=$2 B=$3 T=$4 H=$5 bt=$6 nw=$7 k=$8
  echo ""
  echo "################################################################################"
  echo "# ${label}"
  echo "#   which=${which} B=${B} T=${T} H=${H} bt=${bt} nw=${nw}  kernel=${k}"
  echo "################################################################################"
  ncu -k "regex:${k}" -c 1 --launch-skip 3 $SECTIONS --metrics "$METRICS" \
      python "$DRIVER" --which "$which" --B "$B" --T "$T" --H "$H" --bt "$bt" --nw "$nw" 2>&1 \
      | tag | grep -vE '^DONE'
}

echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "ncu: $(ncu --version | grep -i 'version' | head -1)"
echo "torch: $(python -c 'import torch; print(torch.__version__)' 2>/dev/null)"

run "B2 wy_dqkg bf16 @BT=16 (deployed, nw=4)"  b2 8 2048 8 16 4 _wy_dqkg_kernel
run "B2 wy_dqkg bf16 @BT=64 (nw=4)"            b2 8 2048 8 64 4 _wy_dqkg_kernel
run "B3 intra  bf16 @BT=16 (deployed, nw=4)"   b3 8 2048 8 16 4 _intra_kernel
run "B3 intra  bf16 @BT=16 (nw=2 best)"        b3 8 2048 8 16 2 _intra_kernel
run "B3 intra  bf16 @BT=64 (nw=4)"             b3 8 2048 8 64 4 _intra_kernel
run "B2 wy_dqkg bf16 small @BT=16 (T512 H4)"   b2 1 512  4 16 4 _wy_dqkg_kernel
run "B3 intra  bf16 small @BT=16 (T512 H4)"    b3 1 512  4 16 4 _intra_kernel

echo ""
echo "DONE"
