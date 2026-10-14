#!/usr/bin/env bash
# P7-104: `ncu` side -- attribute the production six-segment bf16 native-bf16-mma **TMA** build
# (`bf16mma_tma`) as **K** moves (not only V).  Section 89 (P7-86) profiled `bf16mma_tma` with
# **K=128 fixed, V in {64,128,256}**; section 94 (P7-91) profiled `bf16`/`bf16mma` with K in
# {64,128,256}.  The K-axis TMA cross product was never profiled -- the exact gap left verbatim
# by section 106.6 (P7-103): "仍未做：bf16mma_tma 的 ncu" (and by section 93.6: "TMA ... x K!=128
# 未做 ncu").  This script closes it.
#
# Tags (`K=V`-generic production launch, all `-DB2M_PROD -DB2M_HDW`):
#   prod          fp32 / cp.async / BVK=32      (fp32 anchor)
#   bf16mma       bf16 / native bf16 mma / BVK=32 / cp.async   (deploy default; 5 bf16, raw v fp32)
#   bf16mma64     bf16 / native bf16 mma / BVK=64 / cp.async   (BVK control for the TMA delta)
#   bf16mma_tma   bf16 / native bf16 mma / BVK=64 / TMA        (bf16 TMA box -> BVK in {64,128})
#
# Grid: K sweep at V=128 (`(64,128),(128,128),(256,128)`) + V sweep at K=128 (`(128,64),(128,256)`).
# Read the two isolations as:
#   bf16mma64 / bf16mma   -> pure **BVK** (V-reduction grouping)
#   bf16mma_tma/bf16mma64 -> pure **TMA** staging engine (numerically bit-identical, P7-103)
#
# NOTE: nvcc hangs under ncu in this container (P7-16) -> pre-build outside ncu.
# Usage (inside kernel_lab):
#   CUDA_VISIBLE_DEVICES=1 bash .../probe_b2_kvk_ncu.sh > .../probe_b2_kvk_ncu.out.txt 2>&1
set -uo pipefail

DIR=${HOME}/proj/tech_record/code/flash-kda
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-1}
cd "$DIR"
CU="$DIR/harness/b2_mma_probe.cu"

echo "== prebuild (outside ncu) =="
build() {  # tag K V extra...
  tag=$1; K=$2; V=$3; shift 3
  so=/tmp/b2kvk_ncu_${tag}_k${K}_v${V}.so
  nvcc -std=c++17 -O3 -arch=sm_90a -Xcompiler -fPIC -shared -lcuda \
       -DB2M_PROD=1 -DB2M_HDW=1 -DB2M_KD=$K -DB2M_VD=$V "$@" "$CU" -o "$so" \
    && echo "  built ${tag} K=${K} V=${V}" || { echo "  FAIL ${tag} K=${K} V=${V}"; exit 1; }
}
extra() {  # tag
  case $1 in
    prod)         echo "" ;;
    bf16mma)      echo "-DB2M_BF16=1 -DB2M_BF16MMA=1" ;;
    bf16mma64)    echo "-DB2M_BF16=1 -DB2M_BF16MMA=1 -DB2M_BVK=64" ;;
    bf16mma_tma)  echo "-DB2M_BF16=1 -DB2M_BF16MMA=1 -DB2M_TMA=1 -DB2M_BVK=64" ;;
    *) echo "bad tag $1" >&2; exit 1 ;;
  esac
}
TAGS="prod bf16mma bf16mma64 bf16mma_tma"
PAIRS="64:128 128:128 256:128 128:64 128:256"
for kv in $PAIRS; do
  K=${kv%%:*}; V=${kv##*:}
  for tag in $TAGS; do
    build "$tag" "$K" "$V" $(extra "$tag")
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

prof() {  # shape K V tag
  shape=$1; K=$2; V=$3; tag=$4
  echo ""
  echo "################################################################################"
  echo "# b2prod_launch  shape=${shape}  K=${K}  V=${V}  tag=${tag}"
  echo "################################################################################"
  ncu -k "regex:b2mma_kernel" -c 1 --launch-skip 3 --metrics "$M" \
    python "$DIR/harness/ncu_b2_k128_dtype_driver.py" --shape "$shape" --K "$K" --V "$V" \
      --tag "$tag" --so /tmp/b2kvk_ncu_${tag}_k${K}_v${V}.so --reps 6 2>&1 \
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
