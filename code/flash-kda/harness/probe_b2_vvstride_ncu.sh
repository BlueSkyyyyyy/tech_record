#!/usr/bin/env bash
# P7-86 diagnostic: test the section 88.5 hypothesis that the `vdw` (BF16_VDW) build's V=256
# bank-conflict uptick comes from the bf16 raw-`v` (`vv`) staging row-stride phase.  Builds the
# V=256 production kernel with `-DB2M_BF16` (vv fp32, anchor), and with `-DB2M_BF16_VDW` at extra
# `vv` row paddings B2M_VVSTRIDE in {0,8,16,24,32}.  If the conflict count tracks the stride, the
# hypothesis holds; if it is flat, the uptick is not a pure stride-phase artifact.
#
# Usage (inside kernel_lab):
#   bash .../probe_b2_vvstride_ncu.sh > .../probe_b2_vvstride_ncu.out.txt 2>&1
set -uo pipefail
DIR=${HOME}/proj/tech_record/code/flash-kda
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-1}
cd "$DIR"
CU="$DIR/harness/b2_mma_probe.cu"

echo "== prebuild (outside ncu) =="
nvcc -std=c++17 -O3 -arch=sm_90a -Xcompiler -fPIC -shared -lcuda \
  -DB2M_PROD=1 -DB2M_HDW=1 -DB2M_VD=256 -DB2M_BF16=1 "$CU" -o /tmp/b2vv_bf16_v256.so \
  && echo "  built bf16 V=256"
for s in 0 8 16 24 32; do
  nvcc -std=c++17 -O3 -arch=sm_90a -Xcompiler -fPIC -shared -lcuda \
    -DB2M_PROD=1 -DB2M_HDW=1 -DB2M_VD=256 -DB2M_BF16=1 -DB2M_BF16_VDW=1 -DB2M_VVSTRIDE=$s \
    "$CU" -o /tmp/b2vv_s${s}_v256.so && echo "  built vdw s=$s"
done

M="l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum,\
l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum,\
lts__t_bytes.sum,dram__bytes_read.sum,gpu__time_duration.sum,\
launch__registers_per_thread,launch__occupancy_limit_shared_mem,\
launch__occupancy_limit_registers"

prof() {  # label dtype_tag so
  label=$1; dtag=$2; so=$3
  echo ""
  echo "#### V=256  tag=${label} (input dtypes: ${dtag})"
  ncu -k "regex:b2mma_kernel" -c 1 --launch-skip 3 --metrics "$M" \
    python "$DIR/harness/ncu_b2_kv2_driver.py" --shape b8t2048h8 --V 256 --tag "$dtag" \
      --so "$so" --reps 6 2>&1 | sed -e '/^==PROF==/d' -e '/^==WARNING==/d'
}

prof bf16 bf16 /tmp/b2vv_bf16_v256.so
for s in 0 8 16 24 32; do
  prof vdw_s$s vdw /tmp/b2vv_s${s}_v256.so
done
