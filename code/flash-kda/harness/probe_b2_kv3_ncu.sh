#!/usr/bin/env bash
# P7-86: ncu side -- close the two-knob cross products that section 88.5 (P7-85) left unprofiled
# at K != V:
#   bf16tma      -DB2M_PROD -DB2M_HDW -DB2M_BF16 -DB2M_TMA             (BF16 x TMA)
#   bf16mma_tma  -DB2M_PROD -DB2M_HDW -DB2M_BF16 -DB2M_BF16MMA -DB2M_TMA
#   damma_tma    -DB2M_PROD -DB2M_HDW -DB2M_DAMMA -DB2M_TMA
# Plus same-run anchors so nothing is compared across two sessions:
#   prod / prod64 / tma  and the BVK=64 non-TMA bf16 anchor `bf1664`.
# K=128 fixed, V in {64,128,256}.  Key mechanical point: a bf16 TMA box is 64 elements (128 B),
# so the bf16 TMA builds MUST use -DB2M_BVK=64 (section 76); the fp32 TMA builds use BVK=32
# (as in section 88).  Hence the bf16tma / bf16mma_tma deltas are read against `bf1664`/`prod64`
# (same BVK) and not against `prod` (BVK=32).
#
# NOTE: nvcc hangs under ncu in this container (P7-16) -> pre-build outside ncu.
# Usage (inside kernel_lab):
#   bash .../probe_b2_kv3_ncu.sh > .../probe_b2_kv3_ncu.out.txt 2>&1
set -uo pipefail

DIR=~/proj/tech_record/code/flash-kda
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-1}
cd "$DIR"
CU="$DIR/harness/b2_mma_probe.cu"

echo "== prebuild (outside ncu) =="
build() {  # tag V extra...
  tag=$1; V=$2; shift 2
  so=/tmp/b2kv3_${tag}_v${V}.so
  nvcc -std=c++17 -O3 -arch=sm_90a -Xcompiler -fPIC -shared -lcuda \
       -DB2M_PROD=1 -DB2M_HDW=1 -DB2M_VD=$V "$@" "$CU" -o "$so" \
    && echo "  built ${tag} V=${V}" || { echo "  FAIL ${tag} V=${V}"; exit 1; }
}
extra() {  # tag
  case $1 in
    prod)         echo "" ;;
    prod64)       echo "-DB2M_BVK=64" ;;
    tma)          echo "-DB2M_TMA=1" ;;
    damma_tma)    echo "-DB2M_DAMMA=1 -DB2M_TMA=1" ;;
    bf1664)       echo "-DB2M_BF16=1 -DB2M_BVK=64" ;;
    bf16tma)      echo "-DB2M_BF16=1 -DB2M_TMA=1 -DB2M_BVK=64" ;;
    bf16mma_tma)  echo "-DB2M_BF16=1 -DB2M_BF16MMA=1 -DB2M_TMA=1 -DB2M_BVK=64" ;;
    *) echo "bad tag $1" >&2; exit 1 ;;
  esac
}
TAGS="prod prod64 tma damma_tma bf1664 bf16tma bf16mma_tma"
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
    python "$DIR/harness/ncu_b2_kv2_driver.py" --shape "$shape" --V "$V" --tag "$tag" \
      --so /tmp/b2kv3_${tag}_v${V}.so --reps 6 2>&1 \
      | sed -e '/^==PROF==/d' -e '/^==WARNING==/d'
}

for V in 64 128 256; do
  for tag in $TAGS; do
    prof b8t2048h8 "$V" "$tag"
  done
done
