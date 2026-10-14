#!/usr/bin/env bash
# P7-100: close the two boundaries left in docs/07 section 102.7 (P7-99):
#   (1) `K!=V` had only ever been crossed with the *fp32 prod* kernel (section 101 Part B);
#       the `-DB2M_INTRA` (B3 diagonal fold) / `-DB2M_DAMMA` (mma-ified DA) knobs were only
#       measured at K==V.  This script crosses K!=V x INTRA/DAMMA.
#   (2) `V=64` had never been measured on the write path.
# Same three-metric write-path decomposition as sections 99/100/101/102:
#   L1 global store sectors (S_L1) -> L2 write sectors + hit/miss -> dram__bytes_write.
# 判据: S_L1 is the store *instruction flow*; it must be bit-identical across dtype/staging/
# mma-ISA/INTRA/DAMMA/K!=V and track the analytic output volume U (S_L1*32/U = 2 default,
# = 1 with `-DB2M_VECST`).
#
# All builds: `-DB2M_PROD -DB2M_HDW`, BT=16, B8T2048H8.
#
#   == Part A: K!=V x INTRA/DAMMA (both K<V and K>V) ==
#     (K,V)=(64,128):  tag=prod / damma / intra / intra_damma
#     (K,V)=(128,64):  tag=prod / damma / intra / intra_damma
#   == Part B: V=64 sweep (calibrate S_L1*32/U at V=64) + INTRA at V=64 ==
#     (K,V)=(64,64):   tag=prod / intra
#     (K,V)=(256,64):  tag=prod / intra
#     (K,V)=(64,256):  tag=intra               (K<V, K=64)
#
# nvcc hangs under ncu in this container (P7-16) -> pre-build outside ncu.  Idle GPU required.
# Usage (inside kernel_lab, absolute path):
#   bash /ssd/.../harness/probe_b2_write_cross3_ncu.sh > /ssd/.../probe_b2_write_cross3_ncu.out.txt 2>&1
set -uo pipefail

DIR=${HOME}/proj/tech_record/code/flash-kda
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-1}
cd "$DIR"
CU="$DIR/harness/b2_mma_probe.cu"

echo "== prebuild (outside ncu) =="
BASE="-std=c++17 -O3 -arch=sm_90a -Xcompiler -fPIC -shared -lcuda -DB2M_PROD=1 -DB2M_HDW=1"
build() {  # tag K V extra...
  tag=$1; K=$2; V=$3; shift 3
  so=/tmp/b2wc3_${tag}_k${K}_v${V}.so
  nvcc $BASE -DB2M_KD=$K -DB2M_VD=$V "$@" "$CU" -o "$so" 2>/dev/null \
    && echo "  built ${tag} K=${K} V=${V}" || { echo "  FAIL ${tag} K=${K} V=${V}"; exit 1; }
}
# Part A: K!=V x INTRA/DAMMA
for kv in "64 128" "128 64"; do
  set -- $kv; K=$1; V=$2
  build prod       "$K" "$V"
  build damma      "$K" "$V" -DB2M_DAMMA=1
  build intra      "$K" "$V" -DB2M_INTRA=1
  build intra_damma "$K" "$V" -DB2M_INTRA=1 -DB2M_DAMMA=1
done
# Part B: V=64 sweep + INTRA at V=64
build prod  64 64
build intra 64 64  -DB2M_INTRA=1
build prod  256 64
build intra 256 64 -DB2M_INTRA=1
build intra 64 256 -DB2M_INTRA=1

M="dram__bytes_read.sum,dram__bytes_write.sum,\
lts__t_bytes.sum,lts__t_sectors_op_read.sum,lts__t_sectors_op_write.sum,\
lts__t_sectors_op_write_lookup_hit.sum,lts__t_sectors_op_write_lookup_miss.sum,\
l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum,\
l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum,\
l1tex__t_requests_pipe_lsu_mem_global_op_st.sum,\
gpu__time_duration.sum,\
lts__throughput.avg.pct_of_peak_sustained_elapsed,\
dram__throughput.avg.pct_of_peak_sustained_elapsed,\
sm__throughput.avg.pct_of_peak_sustained_elapsed,\
sm__warps_active.avg.pct_of_peak_sustained_active,\
launch__registers_per_thread,\
launch__occupancy_limit_shared_mem,launch__occupancy_limit_registers"

prof() {  # shape K V tag skip
  shape=$1; K=$2; V=$3; tag=$4; skip=${5:-3}
  echo ""
  echo "################################################################################"
  echo "# b2prod_launch  shape=${shape}  K=${K}  V=${V}  tag=${tag}  launch_skip=${skip}"
  echo "################################################################################"
  B2KV_SO=/tmp/b2wc3_${tag}_k${K}_v${V}.so B2KV_K=$K B2KV_V=$V B2KV_TAG=$tag \
    ncu -k "regex:b2mma_kernel" -c 1 --launch-skip "$skip" --metrics "$M" \
    python "$DIR/harness/ncu_b2_kv_driver.py" --shape "$shape" --reps 6 2>&1 \
      | sed -e '/^==PROF==/d' -e '/^==WARNING==/d'
}

echo ""
echo "=================== Part A: K!=V x INTRA/DAMMA, K<V ==================="
for tag in prod damma intra intra_damma; do
  prof b8t2048h8 64 128 "$tag"
done

echo ""
echo "=================== Part A: K!=V x INTRA/DAMMA, K>V ==================="
for tag in prod damma intra intra_damma; do
  prof b8t2048h8 128 64 "$tag"
done

echo ""
echo "=================== Part B: V=64 sweep + INTRA at V=64 ==================="
prof b8t2048h8 64 64 prod
prof b8t2048h8 64 64 intra
prof b8t2048h8 256 64 prod
prof b8t2048h8 256 64 intra
prof b8t2048h8 64 256 intra

echo ""
echo "DONE"
