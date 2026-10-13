#!/usr/bin/env bash
# P7-43: scheduler/warp-state ncu pass on the *fused* B2+B3 kernel, to nail the
# "latency-bound" claim behind §46 (the deployed fused kernel sits at 1 block/SM,
# 12.5% occupancy, DRAM 55.8% / tensor 6.2% SOL -> the question is whether the
# issue slots are starved by memory latency, i.e. whether a latency-hiding K2-shape
# restructure has ~2x of headroom).
#
# Usage (inside kernel_lab, idle card):
#   CUDA_VISIBLE_DEVICES=2 bash .../probe_b2b3_bound_ncu.sh > .../probe_b2b3_bound_ncu.out.txt 2>&1

set -uo pipefail

DIR=~/proj/tech_record/code/flash-kda
DRIVER=$DIR/harness/ncu_b2b3_fused_driver.py
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-2}
cd "$DIR"

SECTIONS="--section SchedulerStats --section WarpStateStats --section SpeedOfLight \
--section Occupancy"

METRICS="smsp__warps_eligible.avg.per_cycle_active,\
smsp__issue_active.avg.per_cycle_active,\
smsp__issue_active.avg.pct_of_peak_sustained_active,\
smsp__average_warps_active_per_issue_active.ratio,\
smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio,\
smsp__average_warps_issue_stalled_short_scoreboard_per_issue_active.ratio,\
smsp__average_warps_issue_stalled_wait_per_issue_active.ratio,\
smsp__average_warps_issue_stalled_not_selected_per_issue_active.ratio,\
sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active,\
dram__throughput.avg.pct_of_peak_sustained_elapsed,\
lts__throughput.avg.pct_of_peak_sustained_elapsed"

tag() { sed -e '/^==PROF==/d' -e '/^==WARNING==/d'; }

run() {  # label which B T H bt nw k
  local label=$1 which=$2 B=$3 T=$4 H=$5 bt=$6 nw=$7 k=$8
  echo ""
  echo "################################################################################"
  echo "# ${label}"
  echo "#   which=${which} B=${B} T=${T} H=${H} bt=${bt} nw=${nw}  kernel=${k}"
  echo "################################################################################"
  ncu -k "regex:${k}" -c 1 --launch-skip 3 $SECTIONS --metrics "$METRICS" \
      python "$DRIVER" --which "$which" --B "$B" --T "$T" --H "$H" --bt "$bt" --nw "$nw" \
      2>&1 | tag | grep -vE '^DONE'
}

echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "ncu: $(ncu --version | grep -i 'version' | head -1)"
echo "torch: $(python -c 'import torch; print(torch.__version__)' 2>/dev/null)"

run "FUSED B2+B3 fp32 @BT=16 (nw=8, BV=128, deploy)" fused 8 2048 8 16 8 _fused_b2b3_kernel

echo ""
echo "DONE"
