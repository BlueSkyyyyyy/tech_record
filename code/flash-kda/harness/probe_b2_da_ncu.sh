#!/usr/bin/env bash
# P7-54: ncu side of the CUDA B2 dA-prewich segment.  Profiles the GEMM-core baseline
# (`-DB2M_TMA`, sym b2mma_launch) and the dA build (`-DB2M_DA -DB2M_TMA`, sym
# b2da_launch) on the same operands so the *delta* of the added
# `dApre = dv@v^T + dw@kg^T` segment is read off a single kernel launch each: bytes
# (lts/dram), shared-load traffic + bank conflicts, tensor/SM/DRAM SOL, duration, and the
# SASS LDS/STG counts.
#
# NOTE: nvcc hangs under ncu in this container (P7-16) -> pre-build outside ncu.
# Usage (inside kernel_lab):
#   bash .../probe_b2_da_ncu.sh > .../probe_b2_da_ncu.out.txt 2>&1
set -uo pipefail

DIR=${HOME}/proj/tech_record/code/flash-kda
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-2}
cd "$DIR"

echo "== prebuild (outside ncu) =="
build() {  # $1 = tag, $2.. = -D flags
    local tag="$1"; shift
    nvcc -std=c++17 -O3 -arch=sm_90a -Xcompiler -fPIC -shared -lcuda "$@" \
        "$DIR/harness/b2_mma_probe.cu" -o "/tmp/b2_${tag}.so" 2>/dev/null \
        && echo "  built ${tag}" || { echo "  FAIL ${tag}"; exit 1; }
}
build core_tma -DB2M_TMA=1
build da_tma -DB2M_DA=1 -DB2M_TMA=1

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
sm__warps_active.avg.pct_of_peak_sustained_active"

for pair in "core_tma:ours_core" "da_tma:ours_da"; do
  tag="${pair%%:*}"; mode="${pair##*:}"
  echo ""
  echo "################################################################################"
  echo "# b2mma_kernel variant=${tag}  mode=${mode}  b8t2048h8"
  echo "################################################################################"
  B2D_SO="/tmp/b2_${tag}.so" ncu -k "regex:b2mma_kernel" -c 1 --launch-skip 3 \
      --metrics "$M" \
      python "$DIR/harness/ncu_b2_da_driver.py" --mode "$mode" --shape b8t2048h8 --reps 6 2>&1 \
      | sed -e '/^==PROF==/d' -e '/^==WARNING==/d' \
      | grep -E 'lts__|dram__|l1tex__|gpu__|sm__'
done

echo ""
echo "== SASS shared-load / store counts: da_tma vs core_tma =="
for V in core_tma da_tma; do
  cuobjdump -sass "/tmp/b2_${V}.so" 2>/dev/null | sed -n '/b2mma_kernel/,/^$/p' > "/tmp/sass_${V}.s"
  printf "  %-9s LDS=%s  LDSM=%s  STG=%s  LDGSTS/UBLKCP=%s\n" "$V" \
    "$(grep -cE '\bLDS' "/tmp/sass_${V}.s")" \
    "$(grep -c 'LDSM' "/tmp/sass_${V}.s")" \
    "$(grep -cE '\bSTG' "/tmp/sass_${V}.s")" \
    "$(grep -cE 'LDGSTS|UBLKCP|UTMALDG' "/tmp/sass_${V}.s")"
done

echo ""
echo "DONE"
