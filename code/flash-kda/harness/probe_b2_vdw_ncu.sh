#!/usr/bin/env bash
# P7-80: ncu side -- CUDA B2 **production launch** with the raw `v` / `dw` scratch operands staged
# bf16 (`-DB2M_PROD -DB2M_BF16 -DB2M_BF16_VDW`) vs the P7-75 build (`-DB2M_PROD -DB2M_BF16`, where
# `vv`/`dwm` are fp32).  Question: check_bwd_b2_bf16_vdw.py Part D saw the tf32-mma `vdw` build
# regress 1.06-1.52x while the native-bf16-mma variant stayed neutral -- is the regression a
# shared-load bank-conflict / instruction issue from the bf16 (`OPS`) scratch row stride?
#
# NOTE: nvcc hangs under ncu in this container (P7-16) -> pre-build outside ncu.
# Usage (inside kernel_lab):
#   bash .../probe_b2_vdw_ncu.sh > .../probe_b2_vdw_ncu.out.txt 2>&1
set -uo pipefail

DIR=${HOME}/proj/tech_record/code/flash-kda
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-1}
cd "$DIR"

echo "== prebuild (outside ncu) =="
nvcc -std=c++17 -O3 -arch=sm_90a -Xcompiler -fPIC -shared -lcuda -DB2M_PROD=1 -DB2M_BF16=1 \
    "$DIR/harness/b2_mma_probe.cu" -o /tmp/b2_vdw_bf16.so && echo "  built bf16" \
    || { echo "  FAIL"; exit 1; }
nvcc -std=c++17 -O3 -arch=sm_90a -Xcompiler -fPIC -shared -lcuda -DB2M_PROD=1 -DB2M_BF16=1 \
    -DB2M_BF16_VDW=1 \
    "$DIR/harness/b2_mma_probe.cu" -o /tmp/b2_vdw_vdw.so && echo "  built bf16_vdw" \
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
smsp__inst_executed.sum,\
launch__registers_per_thread,\
launch__occupancy_limit_shared_mem"

for shape in b8t2048h8 b2t1024h8; do
  for variant in "bf16 /tmp/b2_vdw_bf16.so 0" "bf16_vdw /tmp/b2_vdw_vdw.so 1"; do
    set -- $variant
    tag=$1; so=$2; q=$3
    echo ""
    echo "################################################################################"
    echo "# b2prod_launch  shape=${shape}  ${tag}"
    echo "################################################################################"
    B2P_SO="$so" B2BP_BF16=1 B2BP_VDW="$q" ncu -k "regex:b2mma_kernel" -c 1 --launch-skip 3 \
        --metrics "$M" \
        python "$DIR/harness/ncu_b2_bf16_prod_driver.py" --shape "$shape" --reps 6 2>&1 \
        | sed -e '/^==PROF==/d' -e '/^==WARNING==/d'
  done
done
