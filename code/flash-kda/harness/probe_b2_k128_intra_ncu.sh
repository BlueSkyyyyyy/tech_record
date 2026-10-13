#!/usr/bin/env bash
# P7-93: ncu side -- pipe attribution of the **`intra` fold x K!=128** (the "other half" of
# docs/07 section 94.7 blank (2); section 95 bit-checked it but attributed nothing).
# Profile the six-segment production launch **without** (`prod`) vs **with** (`prod_intra`,
# `-DB2M_INTRA`) the folded B3 diagonal at `K in {64,128}` (V=K), to see where the fold's extra
# cost lands as K moves (smem staging q/k/g, bank conflicts, exp2f, occupancy, DRAM/L2).
# Shape b8t2048h8 (grid fills the SMs); K=256 is infeasible for the fold (smem, section 95.3).
#
# NOTE: nvcc hangs under ncu in this container (P7-16) -> pre-build outside ncu.
# Usage (inside kernel_lab):
#   bash .../probe_b2_k128_intra_ncu.sh > .../probe_b2_k128_intra_ncu.out.txt 2>&1
set -uo pipefail

DIR=~/proj/tech_record/code/flash-kda
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-3}
cd "$DIR"
CU="$DIR/harness/b2_mma_probe.cu"

echo "== prebuild (outside ncu) =="
for K in 64 128; do
  nvcc -std=c++17 -O3 -arch=sm_90a -Xcompiler -fPIC -shared -lcuda \
       -DB2M_PROD=1 -DB2M_HDW=1 -DB2M_KD=$K -DB2M_VD=$K "$CU" \
       -o /tmp/b2ki_ncu_prod_k${K}_v${K}.so && echo "  built prod K=${K}" \
    || { echo "  FAIL prod K=${K}"; exit 1; }
  nvcc -std=c++17 -O3 -arch=sm_90a -Xcompiler -fPIC -shared -lcuda \
       -DB2M_PROD=1 -DB2M_HDW=1 -DB2M_INTRA=1 -DB2M_KD=$K -DB2M_VD=$K "$CU" \
       -o /tmp/b2ki_ncu_intra_k${K}_v${K}.so && echo "  built intra K=${K}" \
    || { echo "  FAIL intra K=${K}"; exit 1; }
done

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
launch__occupancy_limit_shared_mem,\
launch__occupancy_limit_registers"

for K in 64 128; do
  for which in prod intra; do
    echo ""
    echo "################################################################################"
    echo "# b2prod${which}_launch  shape=b8t2048h8  K=V=${K}  which=${which}"
    echo "################################################################################"
    ncu -k "regex:b2mma_kernel" -c 1 --launch-skip 3 --metrics "$M" \
      python "$DIR/harness/ncu_b2_k128_intra_driver.py" --shape b8t2048h8 --K "$K" --V "$K" \
        --which "$which" --so /tmp/b2ki_ncu_${which}_k${K}_v${K}.so --reps 6 2>&1 \
      | sed -e '/^==PROF==/d' -e '/^==WARNING==/d'
  done
done
