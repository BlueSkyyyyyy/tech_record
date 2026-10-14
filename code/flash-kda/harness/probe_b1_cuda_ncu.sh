#!/usr/bin/env bash
# P7-16: ncu bound analysis of the *hand-written CUDA* B1 (docs/07 §18, P7-15).
#
# §18 showed the bare-wmma CUDA B1 gets only ~2-6% from its `cp.async` prefetch and is
# NOT load-bound, unlike the Triton prototype of §17.  It *hypothesised* the limiter is
# occupancy / issue / synchronisation, but never profiled the kernel.  This script pins
# it down with ncu and compares the stall composition to:
#   * the two `cp.async` depths (NS=1 no overlap vs NS=3 prefetch), and
#   * a B1_SYNC build (cp.async replaced by plain smem stores), and
#   * FLA `dhu` (its native BT=64), the reference it loses 4.3-9.6x to.
#
# Each config is *pre-built outside ncu* (running nvcc under ncu hangs), then profiled
# with `--no-build`.
#
# Usage (inside kernel_lab):
#   bash /ssd/.../harness/probe_b1_cuda_ncu.sh \
#       > /ssd/.../harness/probe_b1_cuda_ncu.out.txt 2>&1

set -uo pipefail

DIR=${HOME}/proj/tech_record/code/flash-kda
DRIVER=$DIR/harness/ncu_b1_cuda_driver.py
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-1}
cd "$DIR"

SECTIONS="--section SpeedOfLight --section Occupancy --section SchedulerStats \
--section WarpStateStats --section LaunchStats --section MemoryWorkloadAnalysis \
--section ComputeWorkloadAnalysis"

METRICS="sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active,\
sm__pipe_fma_cycles_active.avg.pct_of_peak_sustained_active,\
smsp__average_warps_issue_stalled_barrier_per_issue_active.ratio,\
smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio,\
smsp__average_warps_issue_stalled_short_scoreboard_per_issue_active.ratio,\
smsp__average_warps_issue_stalled_wait_per_issue_active.ratio,\
smsp__average_warps_issue_stalled_mio_throttle_per_issue_active.ratio,\
smsp__average_warps_issue_stalled_not_selected_per_issue_active.ratio,\
smsp__average_warps_issue_stalled_lg_throttle_per_issue_active.ratio,\
smsp__average_warps_issue_stalled_drain_per_issue_active.ratio,\
smsp__average_warps_issue_stalled_selected_per_issue_active.ratio,\
smsp__average_warps_issue_stalled_membar_per_issue_active.ratio,\
l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum,\
l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum,\
l1tex__throughput.avg.pct_of_peak_sustained_elapsed,\
lts__throughput.avg.pct_of_peak_sustained_elapsed,\
dram__throughput.avg.pct_of_peak_sustained_elapsed,\
dram__bytes_read.sum,dram__bytes_write.sum,\
sm__throughput.avg.pct_of_peak_sustained_elapsed"

tag() { sed -e '/^==PROF==/d' -e '/^==WARNING==/d'; }

prebuild() {  # B T H bt ns cflags
  B1_CFLAGS="$6" python "$DRIVER" --which ours --B "$1" --T "$2" --H "$3" --bt "$4" \
      --ns "$5" >/dev/null 2>&1
}

run() {  # label which B T H bt ns cflags kernel_regex
  local label=$1 which=$2 B=$3 T=$4 H=$5 bt=$6 ns=$7 cflags=$8 k=$9
  echo ""
  echo "################################################################################"
  echo "# ${label}"
  echo "#   which=${which} B=${B} T=${T} H=${H} bt=${bt} ns=${ns} cflags='${cflags}' kernel=${k}"
  echo "################################################################################"
  if [ "$which" = "ours" ]; then
    prebuild "$B" "$T" "$H" "$bt" "$ns" "$cflags"
    ncu -k "regex:${k}" -c 1 --launch-skip 3 $SECTIONS --metrics "$METRICS" \
        python "$DRIVER" --which "$which" --B "$B" --T "$T" --H "$H" --bt "$bt" \
        --ns "$ns" --no-build 2>&1 | tag | grep -vE '^DONE'
  else
    ncu -k "regex:${k}" -c 1 --launch-skip 3 $SECTIONS --metrics "$METRICS" \
        python "$DRIVER" --which "$which" --B "$B" --T "$T" --H "$H" --bt "$bt" \
        --ns "$ns" 2>&1 | tag | grep -vE '^DONE'
  fi
}

echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "ncu: $(ncu --version | grep -i 'version' | head -1)"
echo "torch: $(python -c 'import torch; print(torch.__version__)' 2>/dev/null)"

# small shape, isolates the per-chunk serialisation
run "ours-CUDA B1 BT=64 NS=1 (T512 H4)"  ours 1 512 4 64 1 "" b1_dhu_cuda
run "ours-CUDA B1 BT=64 NS=3 (T512 H4)"  ours 1 512 4 64 3 "" b1_dhu_cuda
run "ours-CUDA B1 BT=64 NS=1 SYNC (T512 H4)" ours 1 512 4 64 1 "-DB1_SYNC" b1_dhu_cuda
run "FLA dhu BT=64 (T512 H4)"            fla  1 512 4 64 1 "" bwd_kernel_dhu
# deployed granularity
run "ours-CUDA B1 BT=16 NS=1 (T512 H4)"  ours 1 512 4 16 1 "" b1_dhu_cuda
# big shape: occupancy / grid size effect
run "ours-CUDA B1 BT=64 NS=1 (B8 T2048 H8)" ours 8 2048 8 64 1 "" b1_dhu_cuda
run "FLA dhu BT=64 (B8 T2048 H8)"        fla  8 2048 8 64 1 "" bwd_kernel_dhu

echo ""
echo "DONE"
