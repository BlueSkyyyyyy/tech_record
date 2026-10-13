#!/usr/bin/env bash
# P7-26: ncu byte/occupancy profile of B2 `wy_dqkg` (late-load) under the deployed dtype
# mix, sweeping the two §26.6 items: (1) h/dh dtype, (2) the `i_k`-loop re-read (BK).
# Output: harness/probe_b2_hdh_ncu.out.txt
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
M=dram__bytes_read.sum,dram__bytes_write.sum,lts__t_bytes.sum,gpu__time_duration.sum,launch__registers_per_thread,launch__occupancy_limit_registers,sm__throughput.avg.pct_of_peak_sustained_elapsed,dram__throughput.avg.pct_of_peak_sustained_elapsed,sm__warps_active.avg.pct_of_peak_sustained_active
run() {
  echo "===== $* ====="
  ncu --metrics "$M" -k regex:_wy_dqkg_kernel_late -c 1 --launch-skip 3 \
      python ncu_b2_hdh_driver.py "$@" 2>&1 | grep -E "Metric Value|_wy_dqkg|dram__|lts__|gpu__|launch__|sm__|DONE|Error|error" || true
}
run --hdh fp32 --bk 64
run --hdh bf16 --bk 64
run --hdh fp32 --bk 128
