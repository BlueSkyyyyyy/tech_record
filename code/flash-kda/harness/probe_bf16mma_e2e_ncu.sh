#!/usr/bin/env bash
# P7-79: ncu side of the **bf16 folded production** CUDA B2 (`-DB2M_PROD -DB2M_HDW -DB2M_INTRA
# -DB2M_BF16`) and its native bf16-mma / mma-DA variants.  Goal: see what the ~1-2% chain win of
# `pcibm`/`pcibmd` over `pcib` (docs/07 §82 Part B) is, at the pipe level, and whether the folded
# bf16 core still sits where §78/§81 left the six-segment core (1-2 block/SM, emission-limited).
#
# NOTE: nvcc hangs under ncu in this container (P7-16) -> pre-build outside ncu.
# Usage (inside kernel_lab):
#   bash .../probe_bf16mma_e2e_ncu.sh > .../probe_bf16mma_e2e_ncu.out.txt 2>&1
set -uo pipefail

DIR=${HOME}/proj/tech_record/code/flash-kda
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-1}
cd "$DIR"
CU="$DIR/harness/b2_mma_probe.cu"

echo "== prebuild (outside ncu) =="
nvcc -std=c++17 -O3 -arch=sm_90a -Xcompiler -fPIC -shared -lcuda \
    -DB2M_PROD=1 -DB2M_HDW=1 -DB2M_INTRA=1 -DB2M_BF16=1 \
    "$CU" -o /tmp/b2bm_b_fold.so && echo "  built b_fold" || { echo "  FAIL b_fold"; exit 1; }
nvcc -std=c++17 -O3 -arch=sm_90a -Xcompiler -fPIC -shared -lcuda \
    -DB2M_PROD=1 -DB2M_HDW=1 -DB2M_INTRA=1 -DB2M_BF16=1 -DB2M_BF16MMA=1 \
    "$CU" -o /tmp/b2bm_b_fold_mma.so && echo "  built b_fold_mma" || { echo "  FAIL b_fold_mma"; exit 1; }
nvcc -std=c++17 -O3 -arch=sm_90a -Xcompiler -fPIC -shared -lcuda \
    -DB2M_PROD=1 -DB2M_HDW=1 -DB2M_INTRA=1 -DB2M_BF16=1 -DB2M_BF16MMA=1 -DB2M_DAMMA=1 \
    "$CU" -o /tmp/b2bm_b_fold_mma_da.so && echo "  built b_fold_mma_da" \
    || { echo "  FAIL b_fold_mma_da"; exit 1; }

M="gpu__time_duration.sum,\
lts__t_bytes.sum,lts__throughput.avg.pct_of_peak_sustained_elapsed,\
dram__bytes_read.sum,dram__bytes_write.sum,\
dram__throughput.avg.pct_of_peak_sustained_elapsed,\
sm__throughput.avg.pct_of_peak_sustained_elapsed,\
sm__pipe_tensor_op_hmma_cycles_active.avg.pct_of_peak_sustained_elapsed,\
sm__warps_active.avg.pct_of_peak_sustained_active,\
smsp__average_warps_issue_stalled_wait_per_issue_active.ratio,\
smsp__average_warps_issue_stalled_short_scoreboard_per_issue_active.ratio,\
smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio,\
smsp__average_warps_issue_stalled_barrier_per_issue_active.ratio,\
l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum,\
l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum,\
launch__registers_per_thread,\
launch__shared_mem_per_block_dynamic,\
launch__occupancy_limit_shared_mem,launch__occupancy_limit_registers"

for shape in b8t2048h8 b2t1024h8; do
  for tag in b_fold b_fold_mma b_fold_mma_da; do
    echo ""
    echo "################################################################################"
    echo "# ${tag}  shape=${shape}  (fold, bf16)"
    echo "################################################################################"
    B2BM_SO="/tmp/b2bm_${tag}.so" B2BM_BF16=1 ncu -k "regex:b2mma_kernel" -c 1 --launch-skip 3 \
        --metrics "$M" \
        python "$DIR/harness/ncu_b2_bf16mma_e2e_driver.py" --shape "$shape" --reps 6 2>&1 \
        | sed -e '/^==PROF==/d' -e '/^==WARNING==/d'
  done
done
