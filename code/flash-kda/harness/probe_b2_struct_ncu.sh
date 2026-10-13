#!/usr/bin/env bash
# P7-25: ncu bound analysis for the *late-load* B2 (`wy_dqkg`) rewrite vs the baseline.
# P7-23 (§26) reported B2 at 247 regs/thread -> Block Limit Registers=2 -> 12.4% occ; the
# late-load reordering removes the spills (n_regs 255/12 -> 252/0 at nw=4).  This script
# quantifies what that buys: registers, achieved occupancy, SpeedOfLight, stall mix, traffic.
#
# Usage (inside kernel_lab):
#   bash .../probe_b2_struct_ncu.sh > .../probe_b2_struct_ncu.out.txt 2>&1

set -uo pipefail

DIR=~/proj/tech_record/code/flash-kda
DRIVER=$DIR/harness/ncu_b2_struct_driver.py
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-1}
cd "$DIR"

SECTIONS="--section SpeedOfLight --section Occupancy --section SchedulerStats \
--section WarpStateStats --section LaunchStats --section MemoryWorkloadAnalysis"

METRICS="sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active,\
sm__pipe_fma_cycles_active.avg.pct_of_peak_sustained_active,\
launch__registers_per_thread,launch__occupancy_limit_registers,\
sm__warps_active.avg.pct_of_peak_sustained_active,\
smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio,\
smsp__average_warps_issue_stalled_short_scoreboard_per_issue_active.ratio,\
smsp__average_warps_issue_stalled_wait_per_issue_active.ratio,\
smsp__average_warps_issue_stalled_mio_throttle_per_issue_active.ratio,\
smsp__average_warps_issue_stalled_not_selected_per_issue_active.ratio,\
lts__throughput.avg.pct_of_peak_sustained_elapsed,\
dram__throughput.avg.pct_of_peak_sustained_elapsed,\
dram__bytes_read.sum,dram__bytes_write.sum,lts__t_bytes.sum,\
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

run "B2 baseline  bf16 @BT=16 nw=4 (deployed)" base 8 2048 8 16 4 _wy_dqkg_kernel
run "B2 late-load bf16 @BT=16 nw=4 (rewrite)"  late 8 2048 8 16 4 _wy_dqkg_kernel
run "B2 baseline  bf16 @BT=16 nw=8"            base 8 2048 8 16 8 _wy_dqkg_kernel
run "B2 late-load bf16 @BT=16 nw=8"            late 8 2048 8 16 8 _wy_dqkg_kernel
run "B2 baseline  bf16 @BT=16 small T512H4"    base 1 512  4 16 4 _wy_dqkg_kernel
run "B2 late-load bf16 @BT=16 small T512H4"    late 1 512  4 16 4 _wy_dqkg_kernel

echo ""
echo "DONE"
