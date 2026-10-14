#!/usr/bin/env bash
# K2 occupancy experiment: ncu LaunchStats/Occupancy/SpeedOfLight for the
# baseline (2 CTA/SM) vs cond_in2_out2 (3 CTA/SM) K2, at a grid that fills the
# GPU (B*H=512 CTAs on 132 SMs).
set -uo pipefail
DRIVER=${HOME}/proj/tech_record/code/flash-kda/harness/ncu_driver_occ.py
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-1}

SECTIONS="--section SpeedOfLight --section Occupancy --section LaunchStats"
METRICS="sm__throughput.avg.pct_of_peak_sustained_elapsed,\
lts__throughput.avg.pct_of_peak_sustained_elapsed,\
dram__throughput.avg.pct_of_peak_sustained_elapsed,\
gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed,\
sm__warps_active.avg.pct_of_peak_sustained_active"

run() {  # variant
  local v=$1
  echo ""
  echo "################################################################################"
  echo "# variant=${v}  B=8 T=2048 H=64 (grid_k2=512)"
  echo "################################################################################"
  ncu -k "regex:flash_kda_fwd_recurrence" -c 1 --launch-skip 3 \
      $SECTIONS --metrics "$METRICS" \
      python "$DRIVER" --variant "$v" --B 8 --T 2048 --H 64 2>&1 \
    | sed -e '/^==PROF==/d' -e '/^==WARNING==/d' -e 's/^ *void _flash_kda_fwd_recurrence.*/  KERNEL: _flash_kda_fwd_recurrence (K2)/'
}

echo "ncu: $(ncu --version | grep -i version | head -1)"
run orig_upstream
run plain_in2_out1
run cond_in2_out2
run cond_in2_out1
echo ""
echo "DONE"
