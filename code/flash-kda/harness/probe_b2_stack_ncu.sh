#!/usr/bin/env bash
# P7-106: `ncu` pipe/resource attribution of the **bf16mma_tma multi-knob stack** on the K2-shaped
# CUDA B2 production kernel -- the exact gap left verbatim by section 108.6 (P7-105):
#   "边界：... 未做 ncu（沿用 §107 读侧剖析）、其它两两组合未逐一。"
# P7-105 (docs/07 §108) proved the three knobs (`-DB2M_DAMMA` / `-DB2M_INTRA` / `BF16_VDW`) are
# **value-independent** on the `bmt` base (bit-identical twins).  This script reads the pipe-level
# mechanism: are their **resource / pipe** effects independent when stacked?
#
# Tags (all `-DB2M_PROD -DB2M_HDW -DB2M_BF16 -DB2M_BF16MMA -DB2M_BVK=64 -DB2M_TMA=1` = "bmt"):
#   bmt            base (bf16mma_tma)
#   bmt_da         + -DB2M_DAMMA             (DA -> mma.sync.tf32)
#   bmt_intra      + -DB2M_INTRA             (B3 diagonal folded; extra `dAqk` arg)
#   bmt_da_intra   + -DB2M_DAMMA -DB2M_INTRA (both; the new corner)
#
# `(K,V)` grid: `(128,128)` (square) + `(128,64)` (the worst INTRA regression of §108.3; also the
# only `K!=V` arch where TMA pulls the smem occupancy across the 2-block/SM threshold, §107.4).
# Shape fixed at `B8T2048H8`.
#
# NOTE: nvcc hangs under ncu in this container (P7-16) -> pre-build outside ncu.
# Usage (inside kernel_lab):
#   CUDA_VISIBLE_DEVICES=1 bash .../probe_b2_stack_ncu.sh > .../probe_b2_stack_ncu.out.txt 2>&1
set -uo pipefail

DIR=~/proj/tech_record/code/flash-kda
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-1}
cd "$DIR"
CU="$DIR/harness/b2_mma_probe.cu"
DRV="$DIR/harness/ncu_b2_stack_driver.py"

echo "== prebuild (outside ncu) =="
build() {  # tag K V extra...
  tag=$1; K=$2; V=$3; shift 3
  so=/tmp/b2stk_${tag}_${K}_${V}.so
  nvcc -std=c++17 -O3 -arch=sm_90a -Xcompiler -fPIC -shared -lcuda \
       -DB2M_PROD=1 -DB2M_HDW=1 -DB2M_BF16=1 -DB2M_BF16MMA=1 -DB2M_BVK=64 -DB2M_TMA=1 \
       -DB2M_KD=$K -DB2M_VD=$V "$@" "$CU" -o "$so" \
    && echo "  built ${tag} K=${K} V=${V}" || { echo "  FAIL ${tag} K=${K} V=${V}"; exit 1; }
}
extra() {  # tag
  case $1 in
    bmt)           echo "" ;;
    bmt_da)        echo "-DB2M_DAMMA=1" ;;
    bmt_intra)     echo "-DB2M_INTRA=1" ;;
    bmt_da_intra)  echo "-DB2M_DAMMA=1 -DB2M_INTRA=1" ;;
    *) echo "bad tag $1" >&2; exit 1 ;;
  esac
}
TAGS="bmt bmt_da bmt_intra bmt_da_intra"
PAIRS="128:128 128:64"
for kv in $PAIRS; do
  K=${kv%%:*}; V=${kv##*:}
  for tag in $TAGS; do
    build "$tag" "$K" "$V" $(extra "$tag")
  done
done

M="lts__t_bytes.sum,\
dram__bytes_read.sum,dram__bytes_write.sum,\
l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum,\
l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum,\
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

prof() {  # shape K V tag
  shape=$1; K=$2; V=$3; tag=$4
  echo ""
  echo "################################################################################"
  echo "# b2prod_launch  shape=${shape}  K=${K}  V=${V}  tag=${tag}"
  echo "################################################################################"
  B2STK_SO=/tmp/b2stk_${tag}_${K}_${V}.so B2STK_TAG=$tag B2STK_K=$K B2STK_V=$V \
  ncu -k "regex:b2mma_kernel" -c 1 --launch-skip 3 --metrics "$M" \
    python "$DRV" --shape "$shape" --K "$K" --V "$V" --tag "$tag" \
      --so /tmp/b2stk_${tag}_${K}_${V}.so --reps 6 2>&1 \
      | sed -e '/^==PROF==/d' -e '/^==WARNING==/d'
}

for kv in $PAIRS; do
  K=${kv%%:*}; V=${kv##*:}
  for tag in $TAGS; do
    prof b8t2048h8 "$K" "$V" "$tag"
  done
done
echo ""
echo "== DONE =="
