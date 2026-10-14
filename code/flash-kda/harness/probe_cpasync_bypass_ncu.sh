#!/usr/bin/env bash
# P7-110: ncu collection for `harness/probe_cpasync_bypass.cu`.
#
# Isolates §51.7's open question ("why is cp.async's L1-bypass read path 1.5x
# the DRAM sector count while the identical 16B / identical-pattern plain
# ld.global.v4 is exactly 1.0x?").  For each read instruction (cp.async.cg/ca at
# 16/8/4 B, ld.global[.cg].v4, scalar) and each of the two access patterns we
# report l1tex ld requests/sectors, L2 srcunit read sectors, DRAM read bytes and
# duration -- all over the same 8 MB working set x 4 iters.
#
# NOTE: nvcc hangs under ncu in this container (P7-16) -> pre-build outside ncu.
# Usage (inside kernel_lab):
#   bash .../probe_cpasync_bypass_ncu.sh > .../probe_cpasync_bypass_ncu.out.txt 2>&1
set -uo pipefail

DIR=${HOME}/proj/tech_record/code/flash-kda
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-1}
cd "$DIR"

echo "== build =="
nvcc -std=c++17 -O3 -arch=sm_90a -Xcompiler -fPIC -shared \
    "$DIR/harness/probe_cpasync_bypass.cu" -o /tmp/probe_cpasync_bypass.so 2>&1 \
    && echo "  built" || { echo "  FAIL build"; exit 1; }

M="l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum,\
l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum,\
lts__t_sectors_srcunit_tex_op_read.sum,\
lts__t_sectors_srcunit_tex_op_write.sum,\
lts__t_sectors_op_read.sum,\
lts__t_sectors_op_read_lookup_hit.sum,\
dram__bytes_read.sum,dram__bytes_write.sum,\
lts__t_bytes.sum,gpu__time_duration.sum"

NAMES=(cg16 ca16 ca8 ca4 ld16 ldcg16 ld4)

for PAT in 0 1; do
  if [ "$PAT" = 0 ]; then NUNIT=16384; else NUNIT=4096; fi
  for MODE in 0 1 2 3 4 5 6; do
    echo ""
    echo "################################################################################"
    echo "# probeKernel mode=${MODE}(${NAMES[$MODE]}) pat=${PAT} nunits=${NUNIT} iters=4"
    echo "################################################################################"
    ncu -k "regex:probeKernel" -c 1 --launch-skip 3 --metrics "$M" \
        python "$DIR/harness/probe_cpasync_bypass.py" \
          --mode "$MODE" --pat "$PAT" --nunits "$NUNIT" --iters 4 --reps 5 --no-build 2>&1 \
        | sed -e '/^==PROF==/d' -e '/^==WARNING==/d' -e '/^==PROF==/d'
  done
done
