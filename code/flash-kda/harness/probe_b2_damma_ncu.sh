#!/usr/bin/env bash
# P7-66: ncu side of the DA-segment `mma`-ization (`-DB2M_DAMMA`).  Profiles the CUDA B2
# **production launch** (b2prod_launch, `-DB2M_PROD -DB2M_HDW`) with the scalar (P7-54) vs the
# mma (P7-66) DA segment, so the §68.4 mechanism (DA = scalar shared math, tensor 1.6%,
# bank-conflict x2.2) can be read against the pipe that changes.
#
# NOTE: nvcc hangs under ncu in this container (P7-16) -> pre-build outside ncu.
# Usage (inside kernel_lab, idle card):
#   CUDA_VISIBLE_DEVICES=1 bash .../probe_b2_damma_ncu.sh > .../probe_b2_damma_ncu.out.txt 2>&1
set -uo pipefail

DIR=~/proj/tech_record/code/flash-kda
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-1}
cd "$DIR"

echo "== prebuild (outside ncu) =="
nvcc -std=c++17 -O3 -arch=sm_90a -Xcompiler -fPIC -shared -lcuda \
    -DB2M_PROD=1 -DB2M_HDW=1 "$DIR/harness/b2_mma_probe.cu" -o /tmp/b2_dam_sc.so \
    && echo "  built scalar DA"
nvcc -std=c++17 -O3 -arch=sm_90a -Xcompiler -fPIC -shared -lcuda \
    -DB2M_PROD=1 -DB2M_HDW=1 -DB2M_DAMMA=1 "$DIR/harness/b2_mma_probe.cu" -o /tmp/b2_dam_mma.so \
    && echo "  built mma DA"

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

for v in sc mma; do
  echo ""
  echo "################################################################################"
  echo "# b2prod_launch  DA=$v  shape=b8t2048h8"
  echo "################################################################################"
  B2P_SO="/tmp/b2_dam_$v.so" ncu -k "regex:b2mma_kernel" -c 1 --launch-skip 3 \
      --metrics "$M" \
      python "$DIR/harness/ncu_b2_prod_driver.py" --shape b8t2048h8 --reps 6 2>&1 \
      | sed -e '/^==PROF==/d' -e '/^==WARNING==/d'
done
