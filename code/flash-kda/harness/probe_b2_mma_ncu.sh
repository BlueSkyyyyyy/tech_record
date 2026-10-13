#!/usr/bin/env bash
# P7-46: ncu attribution of the K2-shaped mma CUDA B2 GEMM core vs (a) P7-44's SIMT core
# and (b) the identical-op Triton core, at the deploy shape.  Question: does keeping the
# tensor core + streaming h/dh through smem (no big-operand register residency) move the
# kernel off P7-44's L1/TEX bound and toward Triton's DRAM bound / the floor?
#
# NOTE: nvcc hangs under ncu in this container (P7-16) -> pre-build outside ncu; the
# driver only loads the .so.  Triton path setup launches many kernels -> pin `-k regex`.
#
# Usage (inside kernel_lab):
#   bash .../probe_b2_mma_ncu.sh > .../probe_b2_mma_ncu.out.txt 2>&1

set -uo pipefail

DIR=~/proj/tech_record/code/flash-kda
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-2}
cd "$DIR"

echo "== prebuild (outside ncu) =="
nvcc -std=c++17 -O3 -arch=sm_90a -Xcompiler -fPIC -shared \
    "$DIR/harness/b2_mma_probe.cu" -o /tmp/b2_mma.so || exit 1
nvcc -std=c++17 -O3 -arch=sm_90a -Xcompiler -fPIC -shared \
    "$DIR/harness/b2_core_probe.cu" -o /tmp/b2_core.so || exit 1

SECTIONS="--section SpeedOfLight --section Occupancy --section SchedulerStats \
--section WarpStateStats --section LaunchStats --section MemoryWorkloadAnalysis"

METRICS="sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active,\
sm__pipe_fma_cycles_active.avg.pct_of_peak_sustained_active,\
launch__registers_per_thread,launch__occupancy_limit_registers,\
launch__occupancy_limit_shared_mem,\
sm__warps_active.avg.pct_of_peak_sustained_active,\
smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio,\
smsp__average_warps_issue_stalled_short_scoreboard_per_issue_active.ratio,\
smsp__average_warps_issue_stalled_wait_per_issue_active.ratio,\
smsp__average_warps_issue_stalled_mio_throttle_per_issue_active.ratio,\
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

echo ""
echo "################################################################################"
echo "# K2-shaped mma CUDA B2 core  b8t2048h8  (b2mma_kernel, 256 thr, 8 warps)"
echo "################################################################################"
ncu -k "regex:b2mma_kernel" -c 1 --launch-skip 1 $SECTIONS --metrics "$METRICS" \
    python "$DIR/harness/ncu_b2_mma_driver.py" --mode ours_mma --shape b8t2048h8 --reps 3 2>&1 \
    | tag | grep -vE '^DONE'

echo ""
echo "################################################################################"
echo "# P7-44 SIMT CUDA B2 core  b8t2048h8  (b2_core_kernel, 256 thr)"
echo "################################################################################"
ncu -k "regex:b2_core_kernel" -c 1 --launch-skip 1 $SECTIONS --metrics "$METRICS" \
    python "$DIR/harness/ncu_b2_mma_driver.py" --mode ours_simt --shape b8t2048h8 --reps 3 2>&1 \
    | tag | grep -vE '^DONE'

echo ""
echo "################################################################################"
echo "# Triton identical-op core  b8t2048h8  (_b2core_kernel, nw=8)"
echo "################################################################################"
ncu -k "regex:_b2core_kernel" -c 1 --launch-skip 3 $SECTIONS --metrics "$METRICS" \
    python "$DIR/harness/ncu_b2_mma_driver.py" --mode trit_core --shape b8t2048h8 --reps 8 2>&1 \
    | tag | grep -vE '^DONE'

echo ""
echo "DONE"
