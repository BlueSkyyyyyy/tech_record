#!/usr/bin/env bash
# P7-63: ncu side of the folded B3 `intra` -- profile the production launch **with**
# (`-DB2M_PROD -DB2M_INTRA`) vs **without** (`-DB2M_PROD`) the intra diagonal on the same shape,
# to see where the +~43us on the big shape goes (extra q/k/g + dAqk staging, exp2f, smem).
#
# NOTE: nvcc hangs under ncu in this container (P7-16) -> pre-build outside ncu.
# Usage (inside kernel_lab):
#   bash .../probe_b2_intra_ncu.sh > .../probe_b2_intra_ncu.out.txt 2>&1
set -uo pipefail

DIR=~/proj/tech_record/code/flash-kda
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-1}
cd "$DIR"

echo "== prebuild (outside ncu) =="
nvcc -std=c++17 -O3 -arch=sm_90a -Xcompiler -fPIC -shared -lcuda -DB2M_PROD=1 \
    "$DIR/harness/b2_mma_probe.cu" -o /tmp/b2_intra_prod.so && echo "  built prod" \
    || { echo "  FAIL"; exit 1; }
nvcc -std=c++17 -O3 -arch=sm_90a -Xcompiler -fPIC -shared -lcuda -DB2M_PROD=1 -DB2M_INTRA=1 \
    "$DIR/harness/b2_mma_probe.cu" -o /tmp/b2_intra_fold.so && echo "  built prod_intra" \
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
  for which in prod intra; do
    if [ "$which" = prod ]; then SO=/tmp/b2_intra_prod.so; else SO=/tmp/b2_intra_fold.so; fi
    echo ""
    echo "################################################################################"
    echo "# b2${which}_launch  shape=${shape}  (${SO})"
    echo "################################################################################"
    B2I_SO="$SO" ncu -k "regex:b2mma_kernel" -c 1 --launch-skip 3 \
        --metrics "$M" \
        python "$DIR/harness/ncu_b2_intra_driver.py" --which "$which" --shape "$shape" --reps 6 2>&1 \
        | sed -e '/^==PROF==/d' -e '/^==WARNING==/d'
  done
done
