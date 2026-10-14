#!/usr/bin/env bash
# P7-49: the named "read-side next lever = TMA input" of §51.6/§51.7.  Replaces the
# K2-shaped mma CUDA B2 GEMM core's `cp.async` operand path (whose `.BYPASS` L1-bypass
# §51.4 pinned to a +0.64 GB L2 read amplification) with `cp.async.bulk.tensor` (TMA)
# into a SWIZZLE_128B smem layout with a per-stage `mbarrier` handshake.
#
# Variants (all in `b2_mma_probe.cu`):
#   base       : cp.async + two scalar 32-bit STG.E per output (the §49 baseline)
#   vecst      : cp.async + 64-bit float2 stores
#   tma        : TMA inputs + scalar stores  (read-side change only)
#   tma_vecst  : TMA inputs + float2 stores  (both §51.4 fixes)
#
# NOTE: nvcc hangs under ncu in this container (P7-16) -> pre-build outside ncu.
# Usage (inside kernel_lab):
#   bash .../probe_b2_tma_ncu.sh > .../probe_b2_tma_ncu.out.txt 2>&1
set -uo pipefail

DIR=${HOME}/proj/tech_record/code/flash-kda
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-2}
cd "$DIR"

echo "== prebuild variants (outside ncu) =="
build() {  # $1 = tag, $2.. = -D flags
    local tag="$1"; shift
    nvcc -std=c++17 -O3 -arch=sm_90a -Xcompiler -fPIC -shared -lcuda "$@" \
        "$DIR/harness/b2_mma_probe.cu" -o "/tmp/b2_mma_${tag}.so" 2>/dev/null \
        && echo "  built ${tag}" || { echo "  FAIL ${tag}"; exit 1; }
}
build base
build vecst     -DB2M_VECST=1
build tma       -DB2M_TMA=1
build tma_vecst -DB2M_TMA=1 -DB2M_VECST=1

M="lts__t_bytes.sum,\
lts__t_sectors_srcunit_tex_op_read.sum,lts__t_sectors_srcunit_tex_op_write.sum,\
lts__t_sectors_op_read.sum,lts__t_sectors_op_write.sum,\
dram__bytes_read.sum,dram__bytes_write.sum,\
l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum,\
l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum,\
l1tex__t_requests_pipe_lsu_mem_global_op_st.sum,\
l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum,\
gpu__time_duration.sum,lts__throughput.avg.pct_of_peak_sustained_elapsed,\
dram__throughput.avg.pct_of_peak_sustained_elapsed"

for V in base vecst tma tma_vecst; do
  cp "/tmp/b2_mma_${V}.so" /tmp/b2_mma.so
  echo ""
  echo "################################################################################"
  echo "# b2mma_kernel variant=${V}  b8t2048h8"
  echo "################################################################################"
  ncu -k "regex:b2mma_kernel" -c 1 --launch-skip 3 --metrics "$M" \
      python "$DIR/harness/ncu_b2_mma_driver.py" --mode ours_mma --shape b8t2048h8 --reps 5 2>&1 \
      | sed -e '/^==PROF==/d' -e '/^==WARNING==/d' \
      | grep -E 'lts__|dram__|l1tex__|gpu__'
done

echo ""
echo "== SASS load widths (base uses LDGSTS=cp.async; tma uses UBLKCP=TMA) =="
for V in base tma; do
  cuobjdump -sass "/tmp/b2_mma_${V}.so" 2>/dev/null | sed -n '/b2mma_kernel/,/^$/p' > "/tmp/sass_${V}.s"
  printf "  %-6s LDGSTS=%s  UBLKCP/UTMALDG=%s  STG.E(32b)=%s  STG.E.64=%s\n" "$V" \
    "$(grep -c 'LDGSTS' "/tmp/sass_${V}.s")" \
    "$(grep -cE 'UBLKCP|UTMALDG' "/tmp/sass_${V}.s")" \
    "$(grep -c 'STG.E ' "/tmp/sass_${V}.s")" \
    "$(grep -c 'STG.E.64' "/tmp/sass_${V}.s")"
done

echo ""
echo "DONE"
