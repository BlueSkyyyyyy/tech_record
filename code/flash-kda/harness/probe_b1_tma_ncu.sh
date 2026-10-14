#!/usr/bin/env bash
# P7-20: ncu the TMA + warp-specialized CUDA B1 (`b1_tma_probe.cu`, docs/07 §23) against
# the same-shaped register-resident control (reg BV=64/NW=4) and FLA `dhu`.
#
# §22.5 left the residual (TMA + warp-specialization deep pipeline) as the next lever.
# This run checks whether the TMA producer warp + mbarrier pipeline (a) lowers the `wait`
# stall of the mma warps, (b) raises tensor density, and (c) what it costs in smem /
# occupancy, explaining the measured ~0.81-0.85x at BT=64.
#
# Pre-build the .so outside ncu (running nvcc under ncu hangs), then profile with --no-build.
#
# Usage (inside kernel_lab):
#   bash /ssd/.../harness/probe_b1_tma_ncu.sh \
#       > /ssd/.../harness/probe_b1_tma_ncu.out.txt 2>&1
set -uo pipefail

DIR=${HOME}/proj/tech_record/code/flash-kda
DRIVER=$DIR/harness/ncu_b1_tma_driver.py
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

# prebuild all extension .so once, outside ncu
python "$DRIVER" --which tma   --B 1 --T 512  --H 4 --bt 64 --ns 2 >/dev/null 2>&1
python "$DRIVER" --which tma   --B 8 --T 2048 --H 8 --bt 64 --ns 2 >/dev/null 2>&1
python "$DRIVER" --which etma  --B 1 --T 512  --H 4 --bt 64 --ns 2 >/dev/null 2>&1
python "$DRIVER" --which etma  --B 8 --T 2048 --H 8 --bt 64 --ns 2 >/dev/null 2>&1
python "$DRIVER" --which reg64 --B 1 --T 512  --H 4 --bt 64 --ns 1 >/dev/null 2>&1
python "$DRIVER" --which reg64 --B 8 --T 2048 --H 8 --bt 64 --ns 1 >/dev/null 2>&1

run() {  # label which B T H bt ns kernel_regex
  local label=$1 which=$2 B=$3 T=$4 H=$5 bt=$6 ns=$7 k=$8
  echo ""
  echo "################################################################################"
  echo "# ${label}"
  echo "#   which=${which} B=${B} T=${T} H=${H} bt=${bt} ns=${ns} kernel=${k}"
  echo "################################################################################"
  ncu -k "regex:${k}" -c 1 --launch-skip 3 $SECTIONS --metrics "$METRICS" \
      python "$DRIVER" --which "$which" --B "$B" --T "$T" --H "$H" --bt "$bt" \
      --ns "$ns" --no-build 2>&1 | tag | grep -vE '^DONE'
}

echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "ncu: $(ncu --version | grep -i 'version' | head -1)"

run "REG64 B1 BT=64 NS=1 (T512 H4)"    reg64 1 512  4 64 1 b1_reg_cuda
run "TMA   B1 BT=64 NS=2 (T512 H4)"    tma   1 512  4 64 2 b1_tma_kernel
run "TMAE  B1 BT=64 NS=2 (T512 H4)"    etma  1 512  4 64 2 b1_tma_kernel
run "REG64 B1 BT=64 NS=1 (B8 T2048H8)" reg64 8 2048 8 64 1 b1_reg_cuda
run "TMA   B1 BT=64 NS=1 (B8 T2048H8)" tma   8 2048 8 64 1 b1_tma_kernel
run "TMAE  B1 BT=64 NS=1 (B8 T2048H8)" etma  8 2048 8 64 1 b1_tma_kernel
run "TMA   B1 BT=64 NS=2 (B8 T2048H8)" tma   8 2048 8 64 2 b1_tma_kernel
run "TMAE  B1 BT=64 NS=2 (B8 T2048H8)" etma  8 2048 8 64 2 b1_tma_kernel
run "FLA dhu BT=64 (B8 T2048H8)"       fla   8 2048 8 64 1 bwd_kernel_dhu

echo ""
echo "DONE"
