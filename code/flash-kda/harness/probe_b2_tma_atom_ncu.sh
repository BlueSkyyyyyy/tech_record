#!/usr/bin/env bash
# P7-51: ncu side of the two-128B-swizzle-atom BVK=64 TMA variant.  §53.5 left exactly one
# path open for `TMA + BVK=64`: split each operand tile into NAT=BVK/32 128 B swizzle atoms
# (not SWIZZLE_NONE dense 256 B rows).  This profiles base32/tma32/base64/tma_atom64 to show
# (a) the read-side fix (L2 bytes, srcunit rd, l1tex global rd=0) is still present at BVK=64,
# and (b) the 32x bank-conflict blow-up of SWIZZLE_NONE (§53.4) is gone.
#
# NOTE: nvcc hangs under ncu in this container (P7-16) -> pre-build outside ncu.
# Usage (inside kernel_lab):
#   bash .../probe_b2_tma_atom_ncu.sh > .../probe_b2_tma_atom_ncu.out.txt 2>&1
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
build base32
build tma32    -DB2M_TMA=1
build base64   -DB2M_BVK=64
build atom64   -DB2M_TMA=1 -DB2M_BVK=64

M="lts__t_bytes.sum,\
lts__t_sectors_srcunit_tex_op_read.sum,\
dram__bytes_read.sum,dram__bytes_write.sum,\
l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum,\
l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum,\
l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum,\
l1tex__data_pipe_lsu_wavefronts_mem_shared.sum,\
gpu__time_duration.sum,\
lts__throughput.avg.pct_of_peak_sustained_elapsed,\
dram__throughput.avg.pct_of_peak_sustained_elapsed,\
sm__throughput.avg.pct_of_peak_sustained_elapsed"

for V in base32 tma32 base64 atom64; do
  cp "/tmp/b2_mma_${V}.so" /tmp/b2_mma.so
  echo ""
  echo "################################################################################"
  echo "# b2mma_kernel variant=${V}  b8t2048h8"
  echo "################################################################################"
  ncu -k "regex:b2mma_kernel" -c 1 --launch-skip 3 --metrics "$M" \
      python "$DIR/harness/ncu_b2_mma_driver.py" --mode ours_mma --shape b8t2048h8 --reps 5 2>&1 \
      | sed -e '/^==PROF==/d' -e '/^==WARNING==/d' \
      | grep -E 'lts__|dram__|l1tex__|gpu__|sm__'
done

echo ""
echo "== SASS: base* use LDGSTS (cp.async); tma* use UBLKCP (TMA) =="
for V in base64 atom64; do
  cuobjdump -sass "/tmp/b2_mma_${V}.so" 2>/dev/null | sed -n '/b2mma_kernel/,/^$/p' > "/tmp/sass_${V}.s"
  printf "  %-8s LDGSTS=%s  UBLKCP/UTMALDG=%s  LDS(shared)=%s\n" "$V" \
    "$(grep -c 'LDGSTS' "/tmp/sass_${V}.s")" \
    "$(grep -cE 'UBLKCP|UTMALDG' "/tmp/sass_${V}.s")" \
    "$(grep -cE 'LDS' "/tmp/sass_${V}.s")"
done

echo ""
echo "DONE"
