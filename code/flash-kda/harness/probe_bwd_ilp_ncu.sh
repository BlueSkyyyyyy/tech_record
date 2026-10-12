#!/usr/bin/env bash
# P7-14: ncu bound analysis of the B1 ILP/pipeline variants (docs/07 §17).
#
# §16.3 attributed the 2.14x gap (same BT=64) to a serial dependency chain + no load
# pipeline (`long_scoreboard` 47-55% of issue intervals).  The P7-14 timing sweep
# (`check_bwd_b1_ilp.py`) shows a *loop-pipelined* B1 (tl.range num_stages) alone closes
# the whole gap, while split-K does not.  This script profiles the variants to confirm
# the causal channel: `long_scoreboard` and `warp cyc/issue` should drop when pipelining
# is turned on, and split-K should not move them much.
#
# Usage (inside kernel_lab):
#   bash /ssd/.../harness/probe_bwd_ilp_ncu.sh > /ssd/.../harness/probe_bwd_ilp_ncu.out.txt 2>&1

set -uo pipefail

DIR=~/proj/tech_record/code/flash-kda
DRIVER=$DIR/harness/ncu_bwd_driver.py
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-1}
cd "$DIR"

SECTIONS="--section SpeedOfLight --section Occupancy --section SchedulerStats \
--section WarpStateStats --section LaunchStats"

METRICS="sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active,\
smsp__average_warps_issue_stalled_barrier_per_issue_active.ratio,\
smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio,\
smsp__average_warps_issue_stalled_short_scoreboard_per_issue_active.ratio,\
smsp__average_warps_issue_stalled_wait_per_issue_active.ratio,\
smsp__average_warps_issue_stalled_not_selected_per_issue_active.ratio,\
smsp__average_warps_issue_stalled_drain_per_issue_active.ratio,\
l1tex__throughput.avg.pct_of_peak_sustained_elapsed,\
lts__throughput.avg.pct_of_peak_sustained_elapsed,\
dram__throughput.avg.pct_of_peak_sustained_elapsed,\
sm__throughput.avg.pct_of_peak_sustained_elapsed"

tag() { sed -e '/^==PROF==/d' -e '/^==WARNING==/d'; }

run() {  # label B T H bt variant ns kernel_regex
  local label=$1 B=$2 T=$3 H=$4 bt=$5 variant=$6 ns=$7 k=$8
  echo ""
  echo "################################################################################"
  echo "# ${label}"
  echo "#   B=${B} T=${T} H=${H} bt=${bt} variant=${variant} ns=${ns}  kernel=${k}"
  echo "################################################################################"
  ncu -k "regex:${k}" -c 1 --launch-skip 3 $SECTIONS --metrics "$METRICS" \
      python "$DRIVER" --which ours --B "$B" --T "$T" --H "$H" --bt "$bt" \
      --variant "$variant" --ns "$ns" 2>&1 | tag | grep -vE '^DONE'
}

echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "ncu: $(ncu --version | grep -i 'version' | head -1)"
echo "torch: $(python -c 'import torch; print(torch.__version__)' 2>/dev/null)"

run "ours base  ns=1 BT=64"         2 1024 8 64 base   1 _dhu_kernel_bf16
run "ours pipe  ns=2 BT=64"         2 1024 8 64 pipe   2 _dhu_ilp
run "ours pipe  ns=3 BT=64"         2 1024 8 64 pipe   3 _dhu_ilp
run "ours splitK ns=1 BT=64"        2 1024 8 64 splitk 1 _dhu_ilp
run "ours both  ns=3 BT=64"         2 1024 8 64 both   3 _dhu_ilp
run "ours base  ns=1 BT=16"         2 1024 8 16 base   1 _dhu_kernel_bf16
run "ours pipe  ns=3 BT=16"         2 1024 8 16 pipe   3 _dhu_ilp

echo ""
echo "DONE"
