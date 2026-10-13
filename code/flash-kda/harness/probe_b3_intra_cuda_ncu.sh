#!/usr/bin/env bash
# P7-62: ncu bound analysis for the K2-shape CUDA B3 `intra` (`b3_intra_probe.cu`) vs the
# Triton MMA B3 (`_intra_kernel_mma`) at the deployed CHUNK=16 granularity (B=8 T=2048 H=8).
# Checks where the two sit -- tensor pipe, DRAM/L2, occupancy -- and where the ~1.3-1.45x
# CUDA advantage (Part B) comes from.
#
# Usage (inside kernel_lab):
#   bash .../probe_b3_intra_cuda_ncu.sh > .../probe_b3_intra_cuda_ncu.out.txt 2>&1

set -uo pipefail

DIR=~/proj/tech_record/code/flash-kda
DRIVER=$DIR/harness/ncu_b3_intra_cuda_driver.py
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
      python "$DRIVER" --which "$which" --B "$B" --T "$T" --H "$H" --bt "$bt" --nw "$nw" \
      --no-build 2>&1 \
      | tag | grep -vE '^DONE'
}

echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "ncu: $(ncu --version | grep -i 'version' | head -1)"
echo "torch: $(python -c 'import torch; print(torch.__version__)' 2>/dev/null)"

# pre-build the CUDA probe OUTSIDE ncu (nvcc can hang when launched under ncu, P7-16).
nvcc -std=c++17 -O3 -arch=sm_90a -Xcompiler -fPIC -shared \
    "$DIR/harness/b3_intra_probe.cu" -o /tmp/b3_intra.so || exit 1

run "B3 intra Triton MMA bf16->fp32 @BT=16 nw=4" triton 8 2048 8 16 4 _intra_kernel_mma
run "B3 intra K2 CUDA   fp32 @BT=16 (8w)"       ours   8 2048 8 16 4 b3intra_kernel

echo ""
echo "DONE"
