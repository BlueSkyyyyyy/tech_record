#!/usr/bin/env bash
# P7-74: ncu attribution of the native bf16 `mma.m16n8k16` change for the K2-shaped mma CUDA
# B2 GEMM core (b2_mma_probe.cu `-DB2M_BF16MMA`).  Question: since the P7-73 bf16-staging
# kernel is DRAM/L2 bound at large shapes, does halving the mma instruction count (k16 vs k8)
# move any pipe, or is the timing win (Part B: ~1% large, ~10% small) a pure issue/tensor item?
#
# NOTE: nvcc hangs under ncu in this container (P7-16) -> pre-build outside ncu.
#
# Usage (inside kernel_lab):
#   bash .../probe_b2_bf16mma_ncu.sh > .../probe_b2_bf16mma_ncu.out.txt 2>&1

set -uo pipefail

DIR=${HOME}/proj/tech_record/code/flash-kda
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-3}
cd "$DIR"

echo "== prebuild (outside ncu) =="
nvcc -std=c++17 -O3 -arch=sm_90a -Xcompiler -fPIC -shared \
    "$DIR/harness/b2_mma_probe.cu" -o /tmp/b2_bmm_base.so || exit 1
nvcc -std=c++17 -O3 -arch=sm_90a -Xcompiler -fPIC -shared -DB2M_BF16 \
    "$DIR/harness/b2_mma_probe.cu" -o /tmp/b2_bmm_bf16.so || exit 1
nvcc -std=c++17 -O3 -arch=sm_90a -Xcompiler -fPIC -shared -DB2M_BF16 -DB2M_BF16MMA \
    "$DIR/harness/b2_mma_probe.cu" -o /tmp/b2_bmm_native.so || exit 1

SECTIONS="--section SpeedOfLight --section Occupancy --section SchedulerStats \
--section WarpStateStats --section LaunchStats --section MemoryWorkloadAnalysis"

METRICS="sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active,\
launch__registers_per_thread,\
sm__warps_active.avg.pct_of_peak_sustained_active,\
smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio,\
smsp__average_warps_issue_stalled_short_scoreboard_per_issue_active.ratio,\
smsp__average_warps_issue_stalled_wait_per_issue_active.ratio,\
smsp__inst_executed.sum,\
l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum,\
l1tex__data_pipe_lsu_wavefronts_mem_shared.sum,\
lts__throughput.avg.pct_of_peak_sustained_elapsed,\
dram__throughput.avg.pct_of_peak_sustained_elapsed,\
dram__bytes_read.sum,dram__bytes_write.sum,lts__t_bytes.sum,\
sm__throughput.avg.pct_of_peak_sustained_elapsed"

tag() { sed -e '/^==PROF==/d' -e '/^==WARNING==/d'; }

echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "ncu: $(ncu --version 2>/dev/null | grep -i 'version' | head -1)"
echo "torch: $(python -c 'import torch; print(torch.__version__)' 2>/dev/null)"

for MODE in bf16 bmm; do
  echo ""
  echo "################################################################################"
  echo "# K2-shaped mma CUDA B2 GEMM core  b8t2048h8  -- mode=$MODE"
  echo "################################################################################"
  ncu -k "regex:b2mma_kernel" -c 1 --launch-skip 1 $SECTIONS --metrics "$METRICS" \
      python "$DIR/harness/ncu_b2_bf16mma_driver.py" --mode "$MODE" --shape b8t2048h8 --reps 3 2>&1 \
      | tag
done

echo ""
echo "# K2-shaped mma CUDA B2 GEMM core  t512h4  -- mode=bf16/bmm"
for MODE in bf16 bmm; do
  echo "----------------------------------------------------------------- t512h4 $MODE"
  ncu -k "regex:b2mma_kernel" -c 1 --launch-skip 1 $SECTIONS --metrics "$METRICS" \
      python "$DIR/harness/ncu_b2_bf16mma_driver.py" --mode "$MODE" --shape t512h4 --reps 3 2>&1 \
      | tag
done

echo ""
echo "DONE"
