#!/usr/bin/env bash
# P7-87: ncu side -- confirm the `BF16_VDW x TMA` cross (and its x BF16MMA stack) is a **counter
# no-op** at K != V.  The claim (source-level, b2_mma_probe.cu): `-DB2M_BF16_VDW` has no branch
# reachable under `-DB2M_TMA`, so the two builds must have identical ncu counters.
#   bf16tma            -DB2M_PROD -DB2M_HDW -DB2M_BF16 -DB2M_TMA              (anchor)
#   bf16_vdw_tma       -DB2M_PROD -DB2M_HDW -DB2M_BF16 -DB2M_BF16_VDW -DB2M_TMA
#   bf16mma_tma        -DB2M_PROD -DB2M_HDW -DB2M_BF16 -DB2M_BF16MMA -DB2M_TMA  (anchor)
#   bf16_vdw_mma_tma   ... + -DB2M_BF16_VDW
# All pin -DB2M_BVK=64 (bf16 128 B atom = 64 elements).  K=128, V in {64,128,256}.
#
# NOTE: nvcc hangs under ncu in this container (P7-16) -> pre-build outside ncu.
# Usage (inside kernel_lab):
#   bash .../probe_b2_kv4_ncu.sh > .../probe_b2_kv4_ncu.out.txt 2>&1
set -uo pipefail

DIR=~/proj/tech_record/code/flash-kda
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-1}
cd "$DIR"
CU="$DIR/harness/b2_mma_probe.cu"

echo "== prebuild (outside ncu) =="
build() {  # tag V extra...
  tag=$1; V=$2; shift 2
  so=/tmp/b2kv4_${tag}_v${V}.so
  nvcc -std=c++17 -O3 -arch=sm_90a -Xcompiler -fPIC -shared -lcuda \
       -DB2M_PROD=1 -DB2M_HDW=1 -DB2M_VD=$V "$@" "$CU" -o "$so" \
    && echo "  built ${tag} V=${V}" || { echo "  FAIL ${tag} V=${V}"; exit 1; }
}
extra() {  # tag
  case $1 in
    bf16tma)           echo "-DB2M_BF16=1 -DB2M_TMA=1 -DB2M_BVK=64" ;;
    bf16_vdw_tma)      echo "-DB2M_BF16=1 -DB2M_BF16_VDW=1 -DB2M_TMA=1 -DB2M_BVK=64" ;;
    bf16mma_tma)       echo "-DB2M_BF16=1 -DB2M_BF16MMA=1 -DB2M_TMA=1 -DB2M_BVK=64" ;;
    bf16_vdw_mma_tma)  echo "-DB2M_BF16=1 -DB2M_BF16_VDW=1 -DB2M_BF16MMA=1 -DB2M_TMA=1 -DB2M_BVK=64" ;;
    *) echo "bad tag $1" >&2; exit 1 ;;
  esac
}
TAGS="bf16tma bf16_vdw_tma bf16mma_tma bf16_vdw_mma_tma"
for V in 64 128 256; do
  for tag in $TAGS; do
    build "$tag" "$V" $(extra "$tag")
  done
done

M="lts__t_bytes.sum,\
dram__bytes_read.sum,dram__bytes_write.sum,\
l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum,\
l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum,\
l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum,\
gpu__time_duration.sum,\
lts__throughput.avg.pct_of_peak_sustained_elapsed,\
dram__throughput.avg.pct_of_peak_sustained_elapsed,\
sm__throughput.avg.pct_of_peak_sustained_elapsed,\
sm__pipe_tensor_op_hmma_cycles_active.avg.pct_of_peak_sustained_elapsed,\
sm__warps_active.avg.pct_of_peak_sustained_active,\
launch__registers_per_thread,\
launch__occupancy_limit_shared_mem,\
launch__occupancy_limit_registers"

prof() {  # shape V tag
  shape=$1; V=$2; tag=$3
  echo ""
  echo "################################################################################"
  echo "# b2prod_launch  shape=${shape}  V=${V}  tag=${tag}"
  echo "################################################################################"
  ncu -k "regex:b2mma_kernel" -c 1 --launch-skip 3 --metrics "$M" \
    python "$DIR/harness/ncu_b2_kv2_driver.py" --shape "$shape" --V "$V" --tag bf16tma \
      --so /tmp/b2kv4_${tag}_v${V}.so --reps 6 2>&1 \
      | sed -e '/^==PROF==/d' -e '/^==WARNING==/d'
}

for V in 64 128 256; do
  for tag in $TAGS; do
    prof b8t2048h8 "$V" "$tag"
  done
done
