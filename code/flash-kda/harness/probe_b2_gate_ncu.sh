#!/usr/bin/env bash
# P7-57: ncu side of the CUDA B2 gate chain.  Profiles the `-DB2M_DKGB -DB2M_GATE -DB2M_TMA`
# build against its `-DB2M_NODG` baseline (same build/signature/smem, gate reduction+stores
# skipped) so the *delta* of the added final `dq/dk/dg` gate chain (per-fragment `exp2f` +
# global `q/k/g` reads + an intra-warp `sum_t kdk` reduction + three stores) is read off a
# single kernel launch each: bytes (lts/dram), global-load sectors, shared traffic + bank
# conflicts, XU/exp2 pipe, tensor/SM/DRAM SOL, duration, and SASS MUFU/LDG/STG counts.
#
# NOTE: nvcc hangs under ncu in this container (P7-16) -> pre-build outside ncu.
# Usage (inside kernel_lab):
#   bash .../probe_b2_gate_ncu.sh > .../probe_b2_gate_ncu.out.txt 2>&1
set -uo pipefail

DIR=${HOME}/proj/tech_record/code/flash-kda
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-1}
cd "$DIR"

echo "== prebuild (outside ncu) =="
build() {
    local tag="$1"; shift
    nvcc -std=c++17 -O3 -arch=sm_90a -Xcompiler -fPIC -shared -lcuda "$@" \
        "$DIR/harness/b2_mma_probe.cu" -o "/tmp/b2_${tag}.so" 2>/dev/null \
        && echo "  built ${tag}" || { echo "  FAIL ${tag}"; exit 1; }
}
build gate_tma_nodg -DB2M_DKGB=1 -DB2M_GATE=1 -DB2M_TMA=1 -DB2M_NODG=1
build gate_tma      -DB2M_DKGB=1 -DB2M_GATE=1 -DB2M_TMA=1

M="lts__t_bytes.sum,\
lts__t_sectors_srcunit_tex_op_read.sum,\
dram__bytes_read.sum,dram__bytes_write.sum,\
l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum,\
l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum,\
l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum,\
gpu__time_duration.sum,\
lts__throughput.avg.pct_of_peak_sustained_elapsed,\
dram__throughput.avg.pct_of_peak_sustained_elapsed,\
sm__throughput.avg.pct_of_peak_sustained_elapsed,\
sm__pipe_tensor_op_hmma_cycles_active.avg.pct_of_peak_sustained_elapsed,\
sm__inst_executed_pipe_xu.sum,\
sm__warps_active.avg.pct_of_peak_sustained_active"

for tag in gate_tma_nodg gate_tma; do
  echo ""
  echo "################################################################################"
  echo "# b2mma_kernel variant=${tag}  b8t2048h8"
  echo "################################################################################"
  B2G_SO="/tmp/b2_${tag}.so" ncu -k "regex:b2mma_kernel" -c 1 --launch-skip 3 \
      --metrics "$M" \
      python "$DIR/harness/ncu_b2_gate_driver.py" --shape b8t2048h8 --reps 6 2>&1 \
      | sed -e '/^==PROF==/d' -e '/^==WARNING==/d' \
      | grep -E 'lts__|dram__|l1tex__|gpu__|sm__'
done

echo ""
echo "== SASS: gate_tma vs gate_tma_nodg (per-function MUFU/LDG/STG/LDS counts) =="
for V in gate_tma_nodg gate_tma; do
  cuobjdump -sass "/tmp/b2_${V}.so" 2>/dev/null | sed -n '/b2mma_kernel/,/^$/p' > "/tmp/sass_${V}.s"
  printf "  %-14s MUFU=%s  LDG=%s  STG=%s  LDS/LDSM=%s  LDGSTS/UBLKCP=%s\n" "$V" \
    "$(grep -cE '\bMUFU' "/tmp/sass_${V}.s")" \
    "$(grep -cE '\bLDG' "/tmp/sass_${V}.s")" \
    "$(grep -cE '\bSTG' "/tmp/sass_${V}.s")" \
    "$(grep -cE '\bLDS|LDSM' "/tmp/sass_${V}.s")" \
    "$(grep -cE 'LDGSTS|UBLKCP|UTMALDG' "/tmp/sass_${V}.s")"
done

echo ""
echo "DONE"
