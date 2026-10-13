#!/usr/bin/env bash
# P7-48: localize §49.4.3's "L2 bytes +35% vs the identical-op Triton core"
# (3.03 vs 2.24 GB) in the K2-shaped mma CUDA B2 GEMM core (`b2_mma_probe.cu`),
# the open lead §50.5 left ("L2 字节比 Triton 高 35% ... 与 dgk/冲突均无关").
#
# Controlled byte ablations (all in `b2_mma_probe.cu`):
#   nohdh        : skip h/dh cp.async (zero-fill smem)  -> bytes side of h/dh
#   nodov        : skip do/vn/dv cp.async               -> bytes side of do/vn/dv
#   nohdh_nodov  : skip all operand reads               -> residual (stores only)
#   vecst        : 64-bit (`float2`) stores vs the default two scalar 32-bit `STG.E`
#   noasync      : normal 16B `float4` ld/st instead of `cp.async` (L1-allocating path)
#   noasync_vecst: both
# `noasync*` are traffic-only diagnostics (they break the async pipeline / are racy);
# `vecst` is the deployable, numerically-identical fix.
#
# NOTE: nvcc hangs under ncu in this container (P7-16) -> pre-build outside ncu.
# Usage (inside kernel_lab):
#   bash .../probe_b2_l2_ncu.sh > .../probe_b2_l2_ncu.out.txt 2>&1
set -uo pipefail

DIR=~/proj/tech_record/code/flash-kda
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-2}
cd "$DIR"

echo "== prebuild variants (outside ncu) =="
build() {  # $1 = tag, $2.. = -D flags
    local tag="$1"; shift
    nvcc -std=c++17 -O3 -arch=sm_90a -Xcompiler -fPIC -shared "$@" \
        "$DIR/harness/b2_mma_probe.cu" -o "/tmp/b2_mma_${tag}.so" 2>/dev/null \
        && echo "  built ${tag}" || { echo "  FAIL ${tag}"; exit 1; }
}
build base
build nohdh        -DB2M_NOHDH=1
build nodov        -DB2M_NODOV=1
build nohdh_nodov  -DB2M_NOHDH=1 -DB2M_NODOV=1
build vecst        -DB2M_VECST=1
build noasync      -DB2M_NOASYNC=1
build noasync_vecst -DB2M_NOASYNC=1 -DB2M_VECST=1

M="lts__t_bytes.sum,\
lts__t_sectors_srcunit_tex_op_read.sum,lts__t_sectors_srcunit_tex_op_write.sum,\
lts__t_sectors_op_read.sum,lts__t_sectors_op_write.sum,\
lts__t_sectors_op_read_lookup_hit.sum,\
dram__bytes_read.sum,dram__bytes_write.sum,\
l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum,\
l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum,\
l1tex__t_requests_pipe_lsu_mem_global_op_st.sum,\
l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum,\
gpu__time_duration.sum,lts__throughput.avg.pct_of_peak_sustained_elapsed,\
dram__throughput.avg.pct_of_peak_sustained_elapsed"

for V in base nohdh nodov nohdh_nodov vecst noasync noasync_vecst; do
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
echo "== SASS store/load widths (base vs vecst) =="
for V in base vecst; do
  cuobjdump -sass "/tmp/b2_mma_${V}.so" 2>/dev/null | sed -n '/b2mma_kernel/,/^$/p' > "/tmp/sass_${V}.s"
  printf "  %-6s STG.E(32b)=%s  STG.E.64=%s  LDGSTS=%s\n" "$V" \
    "$(grep -c 'STG.E ' "/tmp/sass_${V}.s")" \
    "$(grep -c 'STG.E.64' "/tmp/sass_${V}.s")" \
    "$(grep -c 'LDGSTS' "/tmp/sass_${V}.s")"
done

echo ""
echo "DONE"
