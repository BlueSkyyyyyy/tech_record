#!/usr/bin/env bash
# P7-75: ncu side -- the CUDA B2 **production launch** with bf16-native operand staging
# (`-DB2M_PROD -DB2M_BF16`) vs the fp32 cp.async prod build, on the deployed large shape.
# Question: does bf16 staging (five operands halved) move this kernel's bottleneck, or does the
# 0.68x CUPTI win of check_bwd_b2_bf16_prod.py Part C come purely from the GEMM core's read
# bytes while the six non-GEMM segments (fp32 scratch + three [B,T,H,K] output writes) dominate?
#
# NOTE: nvcc hangs under ncu in this container (P7-16) -> pre-build outside ncu.
# Usage (inside kernel_lab):
#   bash .../probe_b2_bf16_prod_ncu.sh > .../probe_b2_bf16_prod_ncu.out.txt 2>&1
set -uo pipefail

DIR=${HOME}/proj/tech_record/code/flash-kda
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-1}
cd "$DIR"

echo "== prebuild (outside ncu) =="
nvcc -std=c++17 -O3 -arch=sm_90a -Xcompiler -fPIC -shared -lcuda -DB2M_PROD=1 \
    "$DIR/harness/b2_mma_probe.cu" -o /tmp/b2_bf16prod_prod.so && echo "  built prod (fp32)" \
    || { echo "  FAIL"; exit 1; }
nvcc -std=c++17 -O3 -arch=sm_90a -Xcompiler -fPIC -shared -lcuda -DB2M_PROD=1 -DB2M_BF16=1 \
    "$DIR/harness/b2_mma_probe.cu" -o /tmp/b2_bf16prod_bf16.so && echo "  built prod (bf16)" \
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
  for variant in "fp32 /tmp/b2_bf16prod_prod.so 0" "bf16 /tmp/b2_bf16prod_bf16.so 1"; do
    set -- $variant
    tag=$1; so=$2; q=$3
    echo ""
    echo "################################################################################"
    echo "# b2prod_launch  shape=${shape}  ${tag}"
    echo "################################################################################"
    B2P_SO="$so" B2BP_BF16="$q" ncu -k "regex:b2mma_kernel" -c 1 --launch-skip 3 \
        --metrics "$M" \
        python "$DIR/harness/ncu_b2_bf16_prod_driver.py" --shape "$shape" --reps 6 2>&1 \
        | sed -e '/^==PROF==/d' -e '/^==WARNING==/d'
  done
done
