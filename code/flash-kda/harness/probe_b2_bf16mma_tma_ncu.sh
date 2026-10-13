#!/usr/bin/env bash
# P7-77: ncu attribution of TMA staging on the **native-bf16 mma** K2-shaped mma CUDA B2 GEMM
# core (b2_mma_probe.cu `-DB2M_BF16 -DB2M_BF16MMA [-DB2M_TMA]`).  Question: does TMA still remove
# the `cp.async` `.BYPASS` L2 read amplification once the mma is native bf16, and what is the
# mechanism behind the Part B timing (`bmm_tma64/bmm64`)?
#
# NOTE: nvcc hangs under ncu in this container (P7-16) -> pre-build outside ncu.
#
# Usage (inside kernel_lab):
#   bash .../probe_b2_bf16mma_tma_ncu.sh > .../probe_b2_bf16mma_tma_ncu.out.txt 2>&1

set -uo pipefail

DIR=~/proj/tech_record/code/flash-kda
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-2}
cd "$DIR"

echo "== prebuild (outside ncu) =="
B="$DIR/harness/b2_mma_probe.cu"
nvcc -std=c++17 -O3 -arch=sm_90a -Xcompiler -fPIC -shared -DB2M_BF16=1 -DB2M_BVK=64 "$B" -o /tmp/b2_bmmtma_bf16_64.so || exit 1
nvcc -std=c++17 -O3 -arch=sm_90a -Xcompiler -fPIC -shared -DB2M_BF16=1 -DB2M_BF16MMA=1 -DB2M_BVK=64 "$B" -o /tmp/b2_bmmtma_bmm64.so || exit 1
nvcc -std=c++17 -O3 -arch=sm_90a -Xcompiler -fPIC -shared -DB2M_TMA=1 -DB2M_BF16=1 -DB2M_BVK=64 -lcuda "$B" -o /tmp/b2_bmmtma_bf16_tma64.so || exit 1
nvcc -std=c++17 -O3 -arch=sm_90a -Xcompiler -fPIC -shared -DB2M_TMA=1 -DB2M_BF16=1 -DB2M_BF16MMA=1 -DB2M_BVK=64 -lcuda "$B" -o /tmp/b2_bmmtma_bmm_tma64.so || exit 1

SECTIONS="--section SpeedOfLight --section Occupancy --section SchedulerStats \
--section WarpStateStats --section LaunchStats --section MemoryWorkloadAnalysis"

METRICS="sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active,\
launch__registers_per_thread,launch__occupancy_limit_registers,\
launch__occupancy_limit_shared_mem,\
sm__warps_active.avg.pct_of_peak_sustained_active,\
smsp__inst_executed.sum,\
smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio,\
smsp__average_warps_issue_stalled_short_scoreboard_per_issue_active.ratio,\
smsp__average_warps_issue_stalled_wait_per_issue_active.ratio,\
l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum,\
l1tex__data_pipe_lsu_wavefronts_mem_shared.sum,\
l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum,\
lts__t_sectors_srcunit_tex_op_read.sum,\
lts__throughput.avg.pct_of_peak_sustained_elapsed,\
dram__throughput.avg.pct_of_peak_sustained_elapsed,\
dram__bytes_read.sum,dram__bytes_write.sum,lts__t_bytes.sum,\
sm__throughput.avg.pct_of_peak_sustained_elapsed"

tag() { sed -e '/^==PROF==/d' -e '/^==WARNING==/d'; }

echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "ncu: $(ncu --version 2>/dev/null | grep -i 'version' | head -1)"
echo "torch: $(python -c 'import torch; print(torch.__version__)' 2>/dev/null)"

for MODE in bmm64 bmm_tma64; do
  echo ""
  echo "################################################################################"
  echo "# K2-shaped mma CUDA B2 GEMM core  b8t2048h8  -- mode=$MODE"
  echo "################################################################################"
  ncu -k "regex:b2mma_kernel" -c 1 --launch-skip 1 $SECTIONS --metrics "$METRICS" \
      python "$DIR/harness/ncu_b2_bf16mma_tma_driver.py" --mode "$MODE" --shape b8t2048h8 --reps 3 2>&1 \
      | tag
done

echo ""
echo "# t512h4 -- mode=bmm64 / bmm_tma64"
for MODE in bmm64 bmm_tma64; do
  echo "----------------------------------------------------------------- t512h4 $MODE"
  ncu -k "regex:b2mma_kernel" -c 1 --launch-skip 1 $SECTIONS --metrics "$METRICS" \
      python "$DIR/harness/ncu_b2_bf16mma_tma_driver.py" --mode "$MODE" --shape t512h4 --reps 3 2>&1 \
      | tag
done

echo ""
echo "DONE"
