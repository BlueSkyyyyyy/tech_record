#!/usr/bin/env bash
# P7-56: ncu side of the CUDA B2 `db` segment.  Profiles the `-DB2M_OUT -DB2M_DKGB -DB2M_DB`
# build against its `-DB2M_NODB` baseline (same build/signature/smem, reduction+store skipped) so
# the *delta* of the added per-token write-gate gradient (`db = sum_v dvb*v + sum_k dkgb*kg`, a
# cross-warp column-sum reduction + one global store + `b_A/beta`-free scalar smem reads) is read
# off a single kernel launch each: bytes (lts/dram), shared-load traffic + bank conflicts,
# tensor/SM/DRAM SOL, duration, and the SASS LDS/LDG/STG counts.
#
# NOTE: nvcc hangs under ncu in this container (P7-16) -> pre-build outside ncu.
# Usage (inside kernel_lab):
#   bash .../probe_b2_db_ncu.sh > .../probe_b2_db_ncu.out.txt 2>&1
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
build db_tma_nodb -DB2M_OUT=1 -DB2M_DKGB=1 -DB2M_DB=1 -DB2M_TMA=1 -DB2M_NODB=1
build db_tma      -DB2M_OUT=1 -DB2M_DKGB=1 -DB2M_DB=1 -DB2M_TMA=1

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

for pair in "db_tma_nodb:ours_nodb" "db_tma:ours_db"; do
  tag="${pair%%:*}"; mode="${pair##*:}"
  echo ""
  echo "################################################################################"
  echo "# b2mma_kernel variant=${tag}  mode=${mode}  b8t2048h8"
  echo "################################################################################"
  B2DB_SO="/tmp/b2_${tag}.so" ncu -k "regex:b2mma_kernel" -c 1 --launch-skip 3 \
      --metrics "$M" \
      python "$DIR/harness/ncu_b2_db_driver.py" --mode "$mode" --shape b8t2048h8 --reps 6 2>&1 \
      | sed -e '/^==PROF==/d' -e '/^==WARNING==/d' \
      | grep -E 'lts__|dram__|l1tex__|gpu__|sm__'
done

echo ""
echo "== SASS shared-load / store counts: db_tma vs db_tma_nodb =="
for V in db_tma_nodb db_tma; do
  cuobjdump -sass "/tmp/b2_${V}.so" 2>/dev/null | sed -n '/b2mma_kernel/,/^$/p' > "/tmp/sass_${V}.s"
  printf "  %-12s LDS=%s  LDSM=%s  STG=%s  LDG/%s=%s\n" "$V" \
    "$(grep -cE '\bLDS' "/tmp/sass_${V}.s")" \
    "$(grep -c 'LDSM' "/tmp/sass_${V}.s")" \
    "$(grep -cE '\bSTG' "/tmp/sass_${V}.s")" \
    "LDGSTS" "$(grep -cE 'LDGSTS|UBLKCP|UTMALDG' "/tmp/sass_${V}.s")"
done

echo ""
echo "DONE"
