#!/usr/bin/env bash
# P7-47: ncu attribution for the bank-conflict ablation on the K2-shaped mma B2 core.
# Question: are the 8.81M `l1tex__data_bank_conflicts_..._mem_shared_op_ld` (attributed in
# §49.4.3 to "direct smem reads of the mma fragments") actually the `dgk` column reduction?
#   base   -> /tmp/b2_mma.so as-is
#   nodgk  -> dgk loop removed (diagnostic; dgk output = 0)
#   dgkcf  -> conflict-free dgk reduction
# Each variant is pre-built (nvcc hangs under ncu, P7-16) and profiled separately.
#
# Usage (inside kernel_lab):
#   bash .../probe_b2_dgk_ncu.sh > .../probe_b2_dgk_ncu.out.txt 2>&1

set -uo pipefail

DIR=~/proj/tech_record/code/flash-kda
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-2}
cd "$DIR"

SECTIONS="--section SpeedOfLight --section Occupancy --section SchedulerStats \
--section WarpStateStats --section LaunchStats"
METRICS="sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active,\
launch__registers_per_thread,launch__occupancy_limit_registers,\
launch__occupancy_limit_shared_mem,\
sm__warps_active.avg.pct_of_peak_sustained_active,\
smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio,\
smsp__average_warps_issue_stalled_short_scoreboard_per_issue_active.ratio,\
smsp__average_warps_issue_stalled_wait_per_issue_active.ratio,\
smsp__average_warps_issue_stalled_mio_throttle_per_issue_active.ratio,\
l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum,\
l1tex__data_pipe_lsu_wavefronts_mem_shared.sum,\
lts__throughput.avg.pct_of_peak_sustained_elapsed,\
dram__throughput.avg.pct_of_peak_sustained_elapsed,\
dram__bytes_read.sum,dram__bytes_write.sum,lts__t_bytes.sum,\
sm__throughput.avg.pct_of_peak_sustained_elapsed"

tag() { sed -e '/^==PROF==/d' -e '/^==WARNING==/d'; }

echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "ncu: $(ncu --version 2>/dev/null | grep -i 'version' | head -1)"

run_variant() {
  local tagd="$1"
  echo ""
  echo "################ b2mma_kernel  B8T2048H8  variant=$tagd ################"
  ncu -k "regex:b2mma_kernel" -c 1 --launch-skip 1 $SECTIONS --metrics "$METRICS" \
      python "$DIR/harness/ncu_b2_mma_driver.py" --mode ours_mma --shape b8t2048h8 --reps 3 2>&1 \
      | tag | grep -vE '^DONE'
}

echo "== prebuild =="
nvcc -std=c++17 -O3 -arch=sm_90a -Xcompiler -fPIC -shared "$DIR/harness/b2_mma_probe.cu" -o /tmp/b2_mma.so || exit 1
run_variant base

nvcc -std=c++17 -O3 -arch=sm_90a -Xcompiler -fPIC -shared -DB2M_NODGK=1 "$DIR/harness/b2_mma_probe.cu" -o /tmp/b2_mma.so || exit 1
run_variant nodgk

nvcc -std=c++17 -O3 -arch=sm_90a -Xcompiler -fPIC -shared -DB2M_DGKCF=1 "$DIR/harness/b2_mma_probe.cu" -o /tmp/b2_mma.so || exit 1
run_variant dgkcf

echo ""
echo "DONE"
