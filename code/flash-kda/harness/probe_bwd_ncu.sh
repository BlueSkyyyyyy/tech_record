#!/usr/bin/env bash
# P7-13: ncu bound analysis for the KDA backward state-adjoint kernel B1 (`dhu`).
#
# The backward chain was only ever timed (CUPTI); its hardware bound was never
# measured (P7-2 only profiled the forward K1/K2).  We profile B1 in two forms:
#   * ours  = Triton `_dhu_kernel_bf16` (P7-8/P7-9 bf16-mma clone), BT=16 and BT=64
#   * FLA   = `chunk_gated_delta_rule_bwd_kernel_dhu_blockdim64` (its native BT=64)
# and read SpeedOfLight / Occupancy / stall reasons / memory traffic.
#
# Usage (inside kernel_lab):
#   bash ~/proj/tech_record/code/flash-kda/harness/probe_bwd_ncu.sh \
#       > ~/proj/tech_record/code/flash-kda/harness/probe_bwd_ncu.out.txt 2>&1

set -uo pipefail

DIR=~/proj/tech_record/code/flash-kda
DRIVER=$DIR/harness/ncu_bwd_driver.py
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
smsp__average_warps_issue_stalled_mio_throttle_per_issue_active.ratio,\
smsp__average_warps_issue_stalled_not_selected_per_issue_active.ratio,\
smsp__average_warps_issue_stalled_lg_throttle_per_issue_active.ratio,\
smsp__average_warps_issue_stalled_drain_per_issue_active.ratio,\
smsp__average_warps_issue_stalled_imc_miss_per_issue_active.ratio,\
l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum,\
l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum,\
l1tex__throughput.avg.pct_of_peak_sustained_elapsed,\
lts__throughput.avg.pct_of_peak_sustained_elapsed,\
dram__throughput.avg.pct_of_peak_sustained_elapsed,\
dram__bytes_read.sum,dram__bytes_write.sum,lts__t_bytes.sum,\
sm__throughput.avg.pct_of_peak_sustained_elapsed,\
gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed"

tag() { sed -e '/^==PROF==/d' -e '/^==WARNING==/d'; }

run() {  # label which B T H bt kernel_regex
  local label=$1 which=$2 B=$3 T=$4 H=$5 bt=$6 k=$7
  echo ""
  echo "################################################################################"
  echo "# ${label}"
  echo "#   which=${which} B=${B} T=${T} H=${H} bt=${bt}  kernel=${k}"
  echo "################################################################################"
  ncu -k "regex:${k}" -c 1 --launch-skip 3 $SECTIONS --metrics "$METRICS" \
      python "$DRIVER" --which "$which" --B "$B" --T "$T" --H "$H" --bt "$bt" 2>&1 \
      | tag | grep -vE '^DONE'
}

echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "ncu: $(ncu --version | grep -i 'version' | head -1)"
echo "torch: $(python -c 'import torch; print(torch.__version__)' 2>/dev/null)"

# deployed config: our B1 consumes the CHUNK=16 K1 workspace with native bf16 operands
run "ours B1 bf16 @BT=16 (deployed, matches P7-9)"      ours 2 1024 8 16 _dhu_kernel
run "ours B1 bf16 @BT=64 (isolates the serial-step count)" ours 2 1024 8 64 _dhu_kernel
run "FLA  dhu bf16 @BT=64 (its native granularity)"     fla  2 1024 8 64 bwd_kernel_dhu
run "ours B1 bf16 @BT=16 small (T512 H4)"               ours 1 512 4 16 _dhu_kernel
run "FLA  dhu bf16 @BT=64 small (T512 H4)"              fla  1 512 4 64 bwd_kernel_dhu

echo ""
echo "DONE"
