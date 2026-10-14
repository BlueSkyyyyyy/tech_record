#!/usr/bin/env bash
# P8-17 / docs/07 §115: close §114.5's explicit honest boundary -- §114 established
# (with `harness/probe_l2_write_evict.*`) that inside one ncu window
# `dram__bytes_write` is "the share of dirty L2 lines flushed before the window
# closes", set by the *ordering* of the read stream vs the stores.  §114.5 noted
# it had NOT been re-run under `ncu --replay-mode application` (only the default
# kernel replay).  This script re-runs the exact same microbench/configs under
# BOTH replay modes in one output, so the §114 mechanism can be checked for
# replay robustness (the same check §111/P7-108 did for the bank-conflict probe).
#
# It also records the *read-side complement*: `dram__bytes_read` should be the
# read working set exactly, invariant to the read/write ordering -- unlike the
# write side, read-only data has no eviction-share ambiguity.
#
# NOTE: nvcc hangs under ncu in this container (P7-16) -> pre-build outside ncu.
# Usage (inside kernel_lab):
#   bash .../probe_l2_write_evict_appreplay_ncu.sh \
#        > .../probe_l2_write_evict_appreplay_ncu.out.txt 2>&1
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
dram__bytes_read.sum,dram__bytes_write.sum,\
gpu__time_duration.sum"

run() {  # run <tag> <mode> <outmb> <readmb> <replay> <c>
  local tag=$1 mode=$2 omb=$3 rmb=$4 rep=${5:-kernel} c=${6:-1}
  echo ""
  echo "################################################################################"
  echo "# TAG ${tag}  mode=${mode} outMB=${omb} readMB=${rmb}  (replay=${rep}, ncu -c ${c})"
  echo "################################################################################"
  ncu --replay-mode "$rep" -k "regex:writeKernel" -c "$c" --launch-skip 3 \
      --metrics "$M" \
      python "$DIR/harness/probe_l2_write_evict.py" \
        --mode "$mode" --outmb "$omb" --readmb "$rmb" --reps 6 --no-build 2>&1 \
      | sed -e '/^==PROF==/d' -e '/^==WARNING==/d'
}

echo ""
echo "=== Part A: output 8 MiB (< L2~50 MiB) -- read/write ordering sweep ==="
echo "--- kernel replay (baseline; reproduces §114.2 A1/A2/A3/A4) ---"
run KR_A1_sO8r0_mode0    0 8  0   kernel
run KR_A2_sO8r128_mode1  1 8  128 kernel
run KR_A3_sO8r128_mode2  2 8  128 kernel
run KR_A4_sO8r128_mode3  3 8  128 kernel
echo "--- application replay (the §114.5 boundary) ---"
run AR_A1_sO8r0_mode0    0 8  0   application
run AR_A2_sO8r128_mode1  1 8  128 application
run AR_A3_sO8r128_mode2  2 8  128 application
run AR_A4_sO8r128_mode3  3 8  128 application

echo ""
echo "=== Part B: output > L2 (64 MiB) store-only vs read-then-store, both replays ==="
run KR_B1_sO64r0_mode0   0 64 0   kernel
run KR_B2_sO64r128_mode1 1 64 128 kernel
run AR_B1_sO64r0_mode0   0 64 0   application
run AR_B2_sO64r128_mode1 1 64 128 application

echo ""
echo "=== Part C: read-footprint sweep at fixed ordering (read-then-store, out 8 MiB) ==="
run KR_C1_sO8r32_mode1   1 8  32  kernel
run KR_C2_sO8r64_mode1   1 8  64  kernel
run KR_C3_sO8r256_mode1  1 8  256 kernel
