#!/usr/bin/env bash
# P7-24: ncu bound analysis for the B3 `intra` MMA-diagonal rewrite vs the elementwise
# baseline, at the deployed CHUNK=16 granularity (B=8 T=2048 H=8).  P7-23 measured the
# baseline at tensor 0.05% / SM 82% / L1/TEX 71%; this checks whether MMA-izing the
# diagonal sub-blocks (which at NC=1 is *all* of B3) actually moves work onto the
# TensorCores and what the Duration/regs/occupancy do.
#
# Usage (inside kernel_lab):
#   bash .../probe_b3_mma_ncu.sh > .../probe_b3_mma_ncu.out.txt 2>&1

set -uo pipefail

DIR=~/proj/tech_record/code/flash-kda
DRIVER=$DIR/harness/ncu_b3_mma_driver.py
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
l1tex__throughput.avg.pct_of_peak_sustained_elapsed,\
lts__throughput.avg.pct_of_peak_sustained_elapsed,\
dram__throughput.avg.pct_of_peak_sustained_elapsed,\
sm__throughput.avg.pct_of_peak_sustained_elapsed"

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

run "B3 intra BASELINE bf16 @BT=16 nw=4" base  8 2048 8 16 4 _intra_kernel
run "B3 intra MMA      bf16 @BT=16 nw=4" mma   8 2048 8 16 4 _intra_kernel_mma
run "B3 intra BASELINE bf16 @BT=16 nw=2" base  8 2048 8 16 2 _intra_kernel
run "B3 intra MMA      bf16 @BT=16 nw=2" mma   8 2048 8 16 2 _intra_kernel_mma

echo ""
echo "DONE"
