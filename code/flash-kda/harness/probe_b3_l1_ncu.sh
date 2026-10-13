#!/usr/bin/env bash
# P7-29: ncu bound analysis for the B3 `intra` L1-locality variants at the deployed
# CHUNK=16 granularity (B=8 T=2048 H=8): P7-24 MMA (mma), P7-29 pipelined k-loop
# (loop ns=1 vs ns=2) -- does software-pipelining the k-loop actually cut the
# `long_scoreboard`/raise `tensor%`, and how do DRAM bytes / Duration move?
#
# Usage (inside kernel_lab):
#   bash .../probe_b3_l1_ncu.sh > .../probe_b3_l1_ncu.out.txt 2>&1

set -uo pipefail

DIR=~/proj/tech_record/code/flash-kda
DRIVER=$DIR/harness/ncu_b3_l1_driver.py
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-1}
cd "$DIR"

SECTIONS="--section SpeedOfLight --section Occupancy --section SchedulerStats \
--section WarpStateStats --section LaunchStats --section MemoryWorkloadAnalysis"

METRICS="sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active,\
sm__pipe_fma_cycles_active.avg.pct_of_peak_sustained_active,\
smsp__average_warps_issue_stalled_barrier_per_issue_active.ratio,\
smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio,\
smsp__average_warps_issue_stalled_short_scoreboard_per_issue_active.ratio,\
smsp__average_warps_issue_stalled_wait_per_issue_active.ratio,\
l1tex__throughput.avg.pct_of_peak_sustained_elapsed,\
lts__throughput.avg.pct_of_peak_sustained_elapsed,\
dram__throughput.avg.pct_of_peak_sustained_elapsed,\
sm__throughput.avg.pct_of_peak_sustained_elapsed,\
dram__bytes_read.sum,dram__bytes_write.sum"

tag() { sed -e '/^==PROF==/d' -e '/^==WARNING==/d'; }

run() {  # label which bt nw bk ns kernel_regex
  local label=$1 which=$2 bt=$3 nw=$4 bk=$5 ns=$6 k=$7
  echo ""
  echo "################################################################################"
  echo "# ${label}"
  echo "#   which=${which} bt=${bt} nw=${nw} bk=${bk} ns=${ns}  kernel=${k}"
  echo "################################################################################"
  ncu -k "regex:${k}" -c 1 --launch-skip 3 $SECTIONS --metrics "$METRICS" \
      python "$DRIVER" --which "$which" --B 8 --T 2048 --H 8 --bt "$bt" --nw "$nw" \
      --bk "$bk" --ns "$ns" 2>&1 | tag | grep -vE '^DONE'
}

echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "ncu: $(ncu --version | grep -i 'version' | head -1)"
echo "torch: $(python -c 'import torch; print(torch.__version__)' 2>/dev/null)"

run "B3 MMA      bf16 @BT=16 nw=4" mma  16 4 32 1 _intra_kernel_mma
run "B3 LOOP ns=1 bf16 @BT=16 nw=4 bk=64" loop 16 4 64 1 _intra_kernel_loop
run "B3 LOOP ns=2 bf16 @BT=16 nw=4 bk=64" loop 16 4 64 2 _intra_kernel_loop
run "B3 LOOP ns=3 bf16 @BT=16 nw=4 bk=32" loop 16 4 32 3 _intra_kernel_loop

echo ""
echo "DONE"
