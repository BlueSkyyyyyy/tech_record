#!/usr/bin/env bash
# P7-72: ncu side of the NSTAGE-pinned NAT sweep (BVK 32/64/128 -> NAT 1/2/4).  Closes §54.5:
#   (1) BVK=128 (NAT=4) never tested -> does the read-side fix stay, does bank conflict stay low?
#   (2) attribute atom64's +0.04 GB L2 vs tma32 -> is it the per-atom-boundary L2 re-touch?
# All TMA variants are built with -DB2M_NS=1 so the atom count is not confounded with pipeline
# depth (BVK=128 forces NS=1: one stage = 155.6 KB > half the 227 KB opt-in budget).  tma32ns2
# is the §54 reference.
#
# NOTE: nvcc hangs under ncu in this container (P7-16) -> pre-build outside ncu.
# Usage (inside kernel_lab):
#   CUDA_VISIBLE_DEVICES=1 bash .../probe_b2_tma_atom128_ncu.sh > .../probe_b2_tma_atom128_ncu.out.txt 2>&1
set -uo pipefail

DIR=${HOME}/proj/tech_record/code/flash-kda
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-1}
cd "$DIR"

echo "== prebuild variants (outside ncu) =="
build() {  # $1 = tag, $2.. = -D flags
    local tag="$1"; shift
    nvcc -std=c++17 -O3 -arch=sm_90a -Xcompiler -fPIC -shared -lcuda "$@" \
        "$DIR/harness/b2_mma_probe.cu" -o "/tmp/b2_mma_${tag}.so" 2>/dev/null \
        && echo "  built ${tag}" || { echo "  FAIL ${tag}"; exit 1; }
}
build base32      -DB2M_NS=1
build tma32ns1    -DB2M_TMA=1 -DB2M_NS=1
build atom64ns1   -DB2M_TMA=1 -DB2M_BVK=64  -DB2M_NS=1
build atom128ns1  -DB2M_TMA=1 -DB2M_BVK=128 -DB2M_NS=1
build tma32ns2    -DB2M_TMA=1 -DB2M_NS=2
build atom64ns2   -DB2M_TMA=1 -DB2M_BVK=64  -DB2M_NS=2

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

for V in base32 tma32ns1 atom64ns1 atom128ns1 tma32ns2 atom64ns2; do
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
echo "== SASS: base* use LDGSTS (cp.async); tma* use UBLKCP (TMA).  atom128 => 4 boxes/operand =="
for V in base32 atom64ns1 atom128ns1; do
  cuobjdump -sass "/tmp/b2_mma_${V}.so" 2>/dev/null | sed -n '/b2mma_kernel/,/^$/p' > "/tmp/sass_${V}.s"
  printf "  %-12s LDGSTS=%s  UBLKCP/UTMALDG=%s  LDS(shared)=%s\n" "$V" \
    "$(grep -c 'LDGSTS' "/tmp/sass_${V}.s")" \
    "$(grep -cE 'UBLKCP|UTMALDG' "/tmp/sass_${V}.s")" \
    "$(grep -cE 'LDS' "/tmp/sass_${V}.s")"
done

echo ""
echo "DONE"
