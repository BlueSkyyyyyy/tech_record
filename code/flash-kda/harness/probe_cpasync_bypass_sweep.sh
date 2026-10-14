#!/usr/bin/env bash
# P7-110: two controlled sweeps that pin the mechanism of §51's cp.async read
# amplification (companion to `probe_cpasync_bypass_ncu.sh`).
#
#   Sweep A (reuse factor): fix pattern/working-set, vary `iters` (temporal
#     reuse).  If the amplification is "L1-bypass has no L1 to serve reuse",
#     then srcunit_read should be  data x iters  for `cp.async.cg`, but
#     data x 1 (constant) for `cp.async.ca` and plain `ld.global.v4`.
#   Sweep B (alignment): fix mode/iters, vary the rows-pattern row pitch.
#     If the l1tex over-fetch (16 -> 18 sectors/request) is alignment, it should
#     move identically for `cg` and `ld`; if it were a cp.async property it
#     would only move for `cg`.
#
# NOTE: nvcc hangs under ncu in this container (P7-16) -> pre-build outside ncu.
# Usage (inside kernel_lab):
#   bash .../probe_cpasync_bypass_sweep.sh > .../probe_cpasync_bypass_sweep.out.txt 2>&1
set -uo pipefail

DIR=~/proj/tech_record/code/flash-kda
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-1}
cd "$DIR"

nvcc -std=c++17 -O3 -arch=sm_90a -Xcompiler -fPIC -shared \
    "$DIR/harness/probe_cpasync_bypass.cu" -o /tmp/probe_cpasync_bypass.so \
    && echo "== built ==" || { echo "build FAIL"; exit 1; }

M="l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum,\
l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum,\
lts__t_sectors_srcunit_tex_op_read.sum,\
lts__t_sectors_op_read_lookup_hit.sum,dram__bytes_read.sum,gpu__time_duration.sum"

echo ""
echo "=== Sweep A: reuse factor (pat=0 contiguous, 8MB working set), iters in {1,2,4,8} ==="
for MODE in 0 1 4; do
  for IT in 1 2 4 8; do
    echo "--- mode=${MODE} iters=${IT} ---"
    ncu -k "regex:probeKernel" -c 1 --launch-skip 3 --metrics "$M" \
        python "$DIR/harness/probe_cpasync_bypass.py" --mode "$MODE" --pat 0 \
          --nunits 16384 --iters "$IT" --reps 3 --no-build 2>&1 \
        | grep -E 'sum|mode=' | grep -v '==PROF=='
  done
done

echo ""
echo "=== Sweep B: rows-pattern pitch (pat=1, mode 0=cg vs 4=ld, iters=4, 4096 units) ==="
for MODE in 0 4; do
  for PI in 32 40 68; do
    echo "--- mode=${MODE} pitch=${PI} ---"
    ncu -k "regex:probeKernel" -c 1 --launch-skip 3 --metrics "$M" \
        python "$DIR/harness/probe_cpasync_bypass.py" --mode "$MODE" --pat 1 \
          --nunits 4096 --iters 4 --pitch "$PI" --reps 3 --no-build 2>&1 \
        | grep -E 'sum|mode=' | grep -v '==PROF=='
  done
done
