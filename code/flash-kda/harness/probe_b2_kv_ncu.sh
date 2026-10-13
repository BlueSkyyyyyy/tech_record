#!/usr/bin/env bash
# P7-84: ncu side -- the CUDA B2 **production launch** at K != V (K=128 fixed, V in {64,128,256}).
# Question (docs/07 section 86.5): does the bf16-entry / folded-intra pipe attribution measured at
# K==V==128 (sections 78/79/66) stay put as V moves?  The V-proportional operands (v/vn/do/dv and
# the [B,T,H,V] dv2/dvb outputs) grow with V while the K-axis work (q/k/g, dqg/dkg/dg, dA, intra)
# is V-independent -- so bf16's byte/halving win should *scale* with V and intra's V-independent
# tail should *shrink* in relative terms.
#
# NOTE: nvcc hangs under ncu in this container (P7-16) -> pre-build outside ncu.
# Usage (inside kernel_lab):
#   bash .../probe_b2_kv_ncu.sh > .../probe_b2_kv_ncu.out.txt 2>&1
set -uo pipefail

DIR=~/proj/tech_record/code/flash-kda
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-1}
cd "$DIR"
CU="$DIR/harness/b2_mma_probe.cu"

echo "== prebuild (outside ncu) =="
build() {  # tag V extra...
  tag=$1; V=$2; shift 2
  so=/tmp/b2kv_${tag}_v${V}.so
  nvcc -std=c++17 -O3 -arch=sm_90a -Xcompiler -fPIC -shared -lcuda \
       -DB2M_PROD=1 -DB2M_HDW=1 -DB2M_VD=$V "$@" "$CU" -o "$so" \
    && echo "  built ${tag} V=${V}" || { echo "  FAIL ${tag} V=${V}"; exit 1; }
}
for V in 64 128 256; do
  build prod   $V
  build bf16   $V -DB2M_BF16=1
  build intra  $V -DB2M_INTRA=1
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
  B2KV_SO=/tmp/b2kv_${tag}_v${V}.so B2KV_V=$V B2KV_TAG=$tag \
    ncu -k "regex:b2mma_kernel" -c 1 --launch-skip 3 --metrics "$M" \
    python "$DIR/harness/ncu_b2_kv_driver.py" --shape "$shape" --reps 6 2>&1 \
      | sed -e '/^==PROF==/d' -e '/^==WARNING==/d'
}

for V in 64 128 256; do
  for tag in prod bf16 intra; do
    prof b8t2048h8 "$V" "$tag"
  done
done
for V in 64 256; do
  for tag in prod bf16; do
    prof b2t1024h8 "$V" "$tag"
  done
done
