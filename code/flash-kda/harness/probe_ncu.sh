#!/usr/bin/env bash
# P7-2: ncu bound analysis for FlashKDA K1 (prepare) / K2 (recurrence).
#
# Runs ``ncu`` on ``harness/ncu_driver.py`` for a few shapes, one kernel at a
# time so the output can be labelled unambiguously (the demangled CUTLASS kernel
# names are enormous, so we strip them and rely on the section headers).
#
# Usage (inside kernel_lab):
#   bash ~/proj/tech_record/code/flash-kda/harness/probe_ncu.sh \
#       > ~/proj/tech_record/code/flash-kda/harness/probe_ncu.out.txt 2>&1

set -uo pipefail

DRIVER=~/proj/tech_record/code/flash-kda/harness/ncu_driver.py
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-1}

SECTIONS="--section SpeedOfLight --section Occupancy --section SchedulerStats \
--section WarpStateStats --section LaunchStats --section MemoryWorkloadAnalysis"

METRICS="sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active,\
sm__pipe_fma_cycles_active.avg.pct_of_peak_sustained_active,\
sm__inst_executed_pipe_tensor.sum,\
sm__inst_executed_pipe_lsu.sum,\
smsp__average_warps_issue_stalled_barrier_per_issue_active.ratio,\
smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio,\
smsp__average_warps_issue_stalled_short_scoreboard_per_issue_active.ratio,\
smsp__average_warps_issue_stalled_wait_per_issue_active.ratio,\
smsp__average_warps_issue_stalled_mio_throttle_per_issue_active.ratio,\
smsp__average_warps_issue_stalled_not_selected_per_issue_active.ratio,\
smsp__average_warps_issue_stalled_selected_per_issue_active.ratio,\
l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum,\
l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum,\
l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum,\
l1tex__data_pipe_lsu_wavefronts_mem_shared_op_st.sum,\
l1tex__throughput.avg.pct_of_peak_sustained_elapsed,\
lts__throughput.avg.pct_of_peak_sustained_elapsed,\
dram__throughput.avg.pct_of_peak_sustained_elapsed,\
dram__bytes_read.sum,dram__bytes_write.sum,lts__t_bytes.sum,\
sm__throughput.avg.pct_of_peak_sustained_elapsed,\
gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed"

# The CUTLASS demangled kernel names are printed on their own line (starting
# with "void _flash_kda_fwd_"); replace them with a short tag so the raw log is
# readable.
tag() { sed -e 's/^ *void _flash_kda_fwd_prepare.*/  KERNEL: _flash_kda_fwd_prepare (K1)/' \
             -e 's/^ *void _flash_kda_fwd_recurrence.*/  KERNEL: _flash_kda_fwd_recurrence (K2)/' \
             -e '/^==PROF==/d' -e '/^==WARNING==/d'; }

run() {  # label B T H kernel_regex
  local label=$1 B=$2 T=$3 H=$4 k=$5
  echo ""
  echo "################################################################################"
  echo "# ${label}  B=${B} T=${T} H=${H}  kernel=${k}"
  echo "################################################################################"
  ( cd ~/github/FlashKDA && \
    ncu -k "regex:${k}" -c 1 --launch-skip 3 $SECTIONS --metrics "$METRICS" \
        python "$DRIVER" --B "$B" --T "$T" --H "$H" 2>&1 ) | tag | grep -vE '^DONE'
}

echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "ncu: $(ncu --version | grep -i 'version' | head -1)"
echo "torch: $(python -c 'import torch; print(torch.__version__)' 2>/dev/null)"

run "small/medium (matches docs/07 §3.1)"   1 1024 4  flash_kda_fwd_prepare
run "small/medium (matches docs/07 §3.1)"   1 1024 4  flash_kda_fwd_recurrence
run "design point (matches docs/07 §3.2)"   1 8192 64 flash_kda_fwd_prepare
run "design point (matches docs/07 §3.2)"   1 8192 64 flash_kda_fwd_recurrence
run "K2 grid fills GPU (B*H=256 CTAs)"      4 1024 64 flash_kda_fwd_recurrence

echo ""
echo "DONE"
