#!/usr/bin/env bash
# P7-68: ncu side of the TMA x DAMMA interaction in the CUDA B2 **production launch**
# (b2prod_launch, `-DB2M_PROD -DB2M_HDW`).  P7-67 found TMA slower by 8-12% at production level
# with the *scalar* DA; P7-68 Part B found that with the mma DA (`-DB2M_DAMMA`) the TMA penalty
# vanishes.  This probe reads the four variants against the pipe that changes to explain why.
#
# NOTE: nvcc hangs under ncu in this container (P7-16) -> pre-build outside ncu.
# Usage (inside kernel_lab, idle card):
#   CUDA_VISIBLE_DEVICES=1 bash .../probe_b2_td_ncu.sh > .../probe_b2_td_ncu.out.txt 2>&1
set -uo pipefail

DIR=${HOME}/proj/tech_record/code/flash-kda
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-1}
cd "$DIR"

echo "== prebuild (outside ncu) =="
for spec in "sc::" "sc_tma::-DB2M_TMA=1" "mma::-DB2M_DAMMA=1" "mma_tma::-DB2M_DAMMA=1 -DB2M_TMA=1"; do
  tag=${spec%%::*}; extra=${spec##*::}
  nvcc -std=c++17 -O3 -arch=sm_90a -Xcompiler -fPIC -shared -lcuda \
      -DB2M_PROD=1 -DB2M_HDW=1 $extra "$DIR/harness/b2_mma_probe.cu" -o "/tmp/b2_td_$tag.so" \
      && echo "  built $tag"
done

M="lts__t_bytes.sum,\
dram__bytes_read.sum,dram__bytes_write.sum,\
gpu__time_duration.sum,\
lts__throughput.avg.pct_of_peak_sustained_elapsed,\
dram__throughput.avg.pct_of_peak_sustained_elapsed,\
sm__throughput.avg.pct_of_peak_sustained_elapsed,\
sm__pipe_tensor_op_hmma_cycles_active.avg.pct_of_peak_sustained_elapsed,\
l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum,\
sm__warps_active.avg.pct_of_peak_sustained_active,\
launch__registers_per_thread,\
launch__occupancy_limit_shared_mem"

for v in sc sc_tma mma mma_tma; do
  echo ""
  echo "################################################################################"
  echo "# b2prod_launch  variant=$v  shape=b8t2048h8"
  echo "################################################################################"
  B2P_SO="/tmp/b2_td_$v.so" ncu -k "regex:b2mma_kernel" -c 1 --launch-skip 3 \
      --metrics "$M" \
      python "$DIR/harness/ncu_b2_prod_driver.py" --shape b8t2048h8 --reps 6 2>&1 \
      | sed -e '/^==PROF==/d' -e '/^==WARNING==/d'
done
