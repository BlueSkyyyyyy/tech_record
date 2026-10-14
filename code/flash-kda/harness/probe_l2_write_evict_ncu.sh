#!/usr/bin/env bash
# P7-111: ncu collection for `harness/probe_l2_write_evict.cu`.
#
# Isolates §99.5 / §102.4's open question: inside one ncu window, is
# `dram__bytes_write` the output size, or the fraction of dirty L2 lines flushed
# before the window closes (L2 eviction pressure from a concurrent read stream;
# output working set vs L2 size)?  The store instructions are identical in every
# variant, so `l1tex__..._op_st` (S_L1) must stay put while `dram__bytes_write`
# moves.
#
# NOTE: nvcc hangs under ncu in this container (P7-16) -> pre-build outside ncu.
# Usage (inside kernel_lab):
#   bash .../probe_l2_write_evict_ncu.sh > .../probe_l2_write_evict_ncu.out.txt 2>&1
set -uo pipefail

DIR=~/proj/tech_record/code/flash-kda
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-1}
cd "$DIR"

echo "== build =="
nvcc -std=c++17 -O3 -arch=sm_90a -Xcompiler -fPIC -shared \
    "$DIR/harness/probe_l2_write_evict.cu" -o /tmp/probe_l2_write_evict.so 2>&1 \
    && echo "  built" || { echo "  FAIL build"; exit 1; }

M="l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum,\
l1tex__t_requests_pipe_lsu_mem_global_op_st.sum,\
lts__t_sectors_srcunit_tex_op_write.sum,\
lts__t_sectors_op_write.sum,\
lts__t_sector_op_write_hit_rate.pct,\
lts__t_sectors_op_write_lookup_miss.sum,\
dram__bytes_read.sum,dram__bytes_write.sum,\
lts__t_bytes.sum,gpu__time_duration.sum,\
launch__occupancy_limit_shared_mem"

run() {  # run <tag> <mode> <outmb> <readmb> <c>
  local tag=$1 mode=$2 omb=$3 rmb=$4 c=${5:-1}
  echo ""
  echo "################################################################################"
  echo "# TAG ${tag}  mode=${mode} outMB=${omb} readMB=${rmb}  (ncu -c ${c})"
  echo "################################################################################"
  ncu -k "regex:writeKernel" -c "$c" --launch-skip 3 --metrics "$M" \
      python "$DIR/harness/probe_l2_write_evict.py" \
        --mode "$mode" --outmb "$omb" --readmb "$rmb" --reps 6 --no-build 2>&1 \
      | sed -e '/^==PROF==/d' -e '/^==WARNING==/d'
}

echo ""
echo "=== Part A: output 8 MB fits in H100 L2 (~50 MB), read-pressure sweep (mode = ordering) ==="
run A1_sO8r0_mode0    0 8  0
run A2_sO8r128_mode1  1 8  128
run A3_sO8r128_mode2  2 8  128
run A4_sO8r128_mode3  3 8  128
run A5_sO8r64_mode1   1 8  64
run A6_sO8r256_mode1  1 8  256

echo ""
echo "=== Part B: output exceeds L2 (64 / 96 MB), with and without read pressure ==="
run B1_sO64r0_mode0    0 64  0
run B2_sO64r128_mode1  1 64  128
run B3_sO96r0_mode0    0 96  0

echo ""
echo "=== Part C: same config repeated (cross-replay variability) ==="
run C1_sO8r0_mode0_x3   0 8  0   3
run C2_sO8r128_mode1_x3 1 8  128 3
run C3_sO8r0_mode0_x3b  0 8  0   3
