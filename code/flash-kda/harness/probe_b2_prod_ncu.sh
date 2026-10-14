#!/usr/bin/env bash
# P7-60: ncu side of the CUDA B2 **production launch** (`-DB2M_PROD`, one kernel = GEMM core +
# all six non-GEMM segments).  Reports the standard pipe/utilisation/byte set so we can see
# whether merging the segments keeps the kernel where the isolated rounds left it (DRAM/L2 bound,
# low tensor %, 1 block/SM) — i.e. whether the ~2x timing win of Part C is *launch/core
# amortisation* and not a change of regime.
#
# NOTE: nvcc hangs under ncu in this container (P7-16) -> pre-build outside ncu.
# Usage (inside kernel_lab):
#   bash .../probe_b2_prod_ncu.sh > .../probe_b2_prod_ncu.out.txt 2>&1
set -uo pipefail

DIR=${HOME}/proj/tech_record/code/flash-kda
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-1}
cd "$DIR"

echo "== prebuild (outside ncu) =="
nvcc -std=c++17 -O3 -arch=sm_90a -Xcompiler -fPIC -shared -lcuda -DB2M_PROD=1 -DB2M_TMA=1 \
    "$DIR/harness/b2_mma_probe.cu" -o /tmp/b2_prod_tma.so && echo "  built prod_tma" \
    || { echo "  FAIL"; exit 1; }

M="lts__t_bytes.sum,\
dram__bytes_read.sum,dram__bytes_write.sum,\
l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum,\
l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum,\
l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum,\
gpu__time_duration.sum,\
lts__throughput.avg.pct_of_peak_sustained_elapsed,\
dram__throughput.avg.pct_of_peak_sustained_elapsed,\
sm__throughput.avg.pct_of_peak_sustained_elapsed,\
sm__pipe_tensor_op_hmma_cycles_active.avg.pct_of_peak_sustained_elapsed,\
sm__warps_active.avg.pct_of_peak_sustained_active,\
launch__registers_per_thread,\
launch__occupancy_limit_shared_mem"

for shape in b8t2048h8 b2t1024h8; do
  echo ""
  echo "################################################################################"
  echo "# b2prod_launch  shape=${shape}"
  echo "################################################################################"
  B2P_SO="/tmp/b2_prod_tma.so" ncu -k "regex:b2mma_kernel" -c 1 --launch-skip 3 \
      --metrics "$M" \
      python "$DIR/harness/ncu_b2_prod_driver.py" --shape "$shape" --reps 6 2>&1 \
      | sed -e '/^==PROF==/d' -e '/^==WARNING==/d'
done
