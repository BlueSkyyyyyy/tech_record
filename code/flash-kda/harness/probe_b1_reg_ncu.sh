#!/usr/bin/env bash
# P7-19: ncu the register-resident CUDA B1 (`b1_reg_probe.cu`, docs/07 §22) and pair it
# against the wmma base (`b1_cuda_probe.cu`) and FLA `dhu` in the same session.
#
# §21.6 named the *magnitude* residual to FLA as "每 warp 的有效 mma 密度" (tensor 2.3% vs
# FLA 14%).  This run checks that the register-resident + ldmatrix + mma.sync rewrite moved
# the stall mix (bank conflict / short_sb / short scoreboard) and the tensor density, at the
# same occupancy, explaining the ~2.4x kernel-level speedup.
#
# Pre-build the .so outside ncu (running nvcc under ncu hangs), then profile with --no-build.
#
# Usage (inside kernel_lab):
#   bash /ssd/.../harness/probe_b1_reg_ncu.sh \
#       > /ssd/.../harness/probe_b1_reg_ncu.out.txt 2>&1
set -uo pipefail

DIR=~/proj/tech_record/code/flash-kda
DRIVER=$DIR/harness/ncu_b1_reg_driver.py
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

# prebuild both extension .so once, outside ncu
python "$DRIVER" --which reg  --B 1 --T 512  --H 4 --bt 64 --ns 1 >/dev/null 2>&1
python "$DRIVER" --which reg  --B 8 --T 2048 --H 8 --bt 64 --ns 1 >/dev/null 2>&1
python "$DRIVER" --which base --B 1 --T 512  --H 4 --bt 64 --ns 1 >/dev/null 2>&1
python "$DRIVER" --which base --B 8 --T 2048 --H 8 --bt 64 --ns 1 >/dev/null 2>&1

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

run "REG  B1 BT=64 NS=1 (T512 H4)"    reg  1 512  4 64 1 b1_reg_cuda
run "BASE B1 BT=64 NS=1 (T512 H4)"    base 1 512  4 64 1 b1_dhu_cuda
run "REG  B1 BT=64 NS=1 (B8 T2048H8)" reg  8 2048 8 64 1 b1_reg_cuda
run "BASE B1 BT=64 NS=1 (B8 T2048H8)" base 8 2048 8 64 1 b1_dhu_cuda
run "FLA dhu BT=64 (B8 T2048H8)"      fla  8 2048 8 64 1 bwd_kernel_dhu

echo ""
echo "DONE"
