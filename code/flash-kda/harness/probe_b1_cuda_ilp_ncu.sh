#!/usr/bin/env bash
# P7-18: ncu the mma-ILP variants of the hand-written CUDA B1 (`-DB1_ILP1` / `-DB1_ILP2`,
# docs/07 §21) and pair them against the base of P7-16 in the same session.
#
# §20 measured that killing the shared-load bank conflicts (padding) left the `wait` stall
# and the tensor density untouched -> the magnitude gap is *parallelism / mma density*.
# This run tests whether reordering the mma loops to expose independent accumulator chains
# (`wait` = fixed mma pipeline latency) actually moves the stall mix, before a K2 rewrite.
#
# Each config is pre-built outside ncu (running nvcc under ncu hangs), then profiled with
# `--no-build`.
#
# Usage (inside kernel_lab):
#   bash /ssd/.../harness/probe_b1_cuda_ilp_ncu.sh \
#       > /ssd/.../harness/probe_b1_cuda_ilp_ncu.out.txt 2>&1

set -uo pipefail

DIR=${HOME}/proj/tech_record/code/flash-kda
DRIVER=$DIR/harness/ncu_b1_cuda_driver.py
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

prebuild() {  # B T H bt ns cflags
  B1_CFLAGS="$6" python "$DRIVER" --which ours --B "$1" --T "$2" --H "$3" --bt "$4" \
      --ns "$5" >/dev/null 2>&1
}

run() {  # label B T H bt ns cflags kernel_regex
  local label=$1 B=$2 T=$3 H=$4 bt=$5 ns=$6 cflags=$7 k=$8
  echo ""
  echo "################################################################################"
  echo "# ${label}"
  echo "#   B=${B} T=${T} H=${H} bt=${bt} ns=${ns} cflags='${cflags}' kernel=${k}"
  echo "################################################################################"
  prebuild "$B" "$T" "$H" "$bt" "$ns" "$cflags"
  ncu -k "regex:${k}" -c 1 --launch-skip 3 $SECTIONS --metrics "$METRICS" \
      python "$DRIVER" --which ours --B "$B" --T "$T" --H "$H" --bt "$bt" \
      --ns "$ns" --no-build 2>&1 | tag | grep -vE '^DONE'
}

echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "ncu: $(ncu --version | grep -i 'version' | head -1)"
echo "torch: $(python -c 'import torch; print(torch.__version__)' 2>/dev/null)"

# small shape (per-chunk serialisation, grid 64): base / ilp1 / ilp2
run "ours-CUDA B1 BT=64 NS=1 base  (T512 H4)" 1 512 4 64 1 ""         b1_dhu_cuda
run "ours-CUDA B1 BT=64 NS=1 ILP1  (T512 H4)" 1 512 4 64 1 "-DB1_ILP1" b1_dhu_cuda
run "ours-CUDA B1 BT=64 NS=1 ILP2  (T512 H4)" 1 512 4 64 1 "-DB1_ILP2" b1_dhu_cuda
# big shape (grid fills the device): base / ilp1 / ilp2
run "ours-CUDA B1 BT=64 NS=1 base  (B8 T2048 H8)" 8 2048 8 64 1 ""         b1_dhu_cuda
run "ours-CUDA B1 BT=64 NS=1 ILP1  (B8 T2048 H8)" 8 2048 8 64 1 "-DB1_ILP1" b1_dhu_cuda
run "ours-CUDA B1 BT=64 NS=1 ILP2  (B8 T2048 H8)" 8 2048 8 64 1 "-DB1_ILP2" b1_dhu_cuda

echo ""
echo "DONE"
