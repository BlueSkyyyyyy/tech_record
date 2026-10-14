#!/usr/bin/env bash
# P7-45: ncu bound/traffic per ablation variant of the fused B2/B3 kernel
# (full / no_post / no_lgemm / allcheap).  Confirms the byte traffic is preserved
# across variants (same loads/stores) and locates each segment's bound.
#
# Usage (inside kernel_lab):
#   bash .../probe_b2b3_seg_ncu.sh > .../probe_b2b3_seg_ncu.out.txt 2>&1

set -uo pipefail

DIR=${HOME}/proj/tech_record/code/flash-kda
DRIVER=$DIR/harness/ncu_b2b3_seg_driver.py
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-1}
cd "$DIR"

SECTIONS="--section SpeedOfLight --section Occupancy --section LaunchStats \
--section MemoryWorkloadAnalysis"
METRICS="sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active,\
sm__pipe_fma_cycles_active.avg.pct_of_peak_sustained_active,\
smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio,\
smsp__average_warps_issue_stalled_short_scoreboard_per_issue_active.ratio,\
smsp__average_warps_issue_stalled_wait_per_issue_active.ratio,\
smsp__inst_executed.avg.per_cycle_active,\
l1tex__throughput.avg.pct_of_peak_sustained_elapsed,\
lts__throughput.avg.pct_of_peak_sustained_elapsed,\
dram__throughput.avg.pct_of_peak_sustained_elapsed,\
dram__bytes_read.sum,dram__bytes_write.sum,lts__t_bytes.sum,\
sm__throughput.avg.pct_of_peak_sustained_elapsed"

tag() { sed -e '/^==PROF==/d' -e '/^==WARNING==/d'; }

run() {  # label variant B T H bt nw
  local label=$1 variant=$2 B=$3 T=$4 H=$5 bt=$6 nw=$7
  echo ""
  echo "################################################################################"
  echo "# ${label}"
  echo "#   variant=${variant} B=${B} T=${T} H=${H} bt=${bt} nw=${nw} kernel=_seg_b2b3_kernel"
  echo "################################################################################"
  ncu -k "regex:_seg_b2b3_kernel" -c 1 --launch-skip 3 $SECTIONS --metrics "$METRICS" \
      python "$DRIVER" --variant "$variant" --B "$B" --T "$T" --H "$H" --bt "$bt" --nw "$nw" \
      2>&1 | tag | grep -vE '^DONE'
}

echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "ncu: $(ncu --version | grep -i 'version' | head -1)"
echo "torch: $(python -c 'import torch; print(torch.__version__)' 2>/dev/null)"

run "full     (deployed fused B2/B3, B8T2048H8)" full     8 2048 8 16 8
run "no_hdh   (h/dh reads -> zeros)        " no_hdh   8 2048 8 16 8
run "bf16_hdh (h/dh read as bf16)          " bf16_hdh 8 2048 8 16 8
run "no_qkg   (q/k/g reads -> zeros)       " no_qkg   8 2048 8 16 8
run "no_all   (all big operand reads -> 0) " no_all  8 2048 8 16 8
run "full     small T512 H4                " full     1 512  4 16 8

echo ""
echo "DONE"
