#!/usr/bin/env bash
# P7-97: close the boundary left in docs/07 section 99.7 -- the section 99 write-path decomposition
# (L1 store sectors -> L2 write sectors + hit/miss -> DRAM write bytes) and its new 判据 ("compare
# write volume by L1 global store sectors, NOT dram__bytes_write") were only measured on the
# **non-TMA fp32/bf16** production build.  Section 99.7 explicitly flagged `-DB2M_TMA` / `-DB2M_BF16MMA`
# as "not retested (TMA uses an independent engine, the write-side 口径 differs -- another thread)".
# This probe retests them.
#
# All builds: `-DB2M_PROD -DB2M_HDW`; V in {128,256}.  Kernel outputs are fp32 and (for the same
# operands) bit-identical; the question is whether the L1 store-sector 判据 stays bit-identical and
# where any residual DRAM-write difference sits when TMA input staging / native-bf16 mma are on.
#
#   tag=prod        -DB2M_BVK=32                              (fp32 baseline; section 99)
#   tag=bf16        -DB2M_BVK=32 -DB2M_BF16
#   tag=bf16mma     -DB2M_BVK=32 -DB2M_BF16 -DB2M_BF16MMA
#   tag=prod64      -DB2M_BVK=64                              (fp32, same BVK as the TMA builds)
#   tag=bf16tma     -DB2M_BVK=64 -DB2M_BF16 -DB2M_TMA
#   tag=bf16mmatma  -DB2M_BVK=64 -DB2M_BF16 -DB2M_BF16MMA -DB2M_TMA
#
# nvcc hangs under ncu in this container (P7-16) -> pre-build outside ncu.  Idle GPU required.
# Usage (inside kernel_lab, absolute path):
#   bash /ssd/.../harness/probe_b2_write_knobs_ncu.sh > /ssd/.../probe_b2_write_knobs_ncu.out.txt 2>&1
set -uo pipefail

DIR=${HOME}/proj/tech_record/code/flash-kda
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-1}
cd "$DIR"
CU="$DIR/harness/b2_mma_probe.cu"

echo "== prebuild (outside ncu) =="
BASE="-std=c++17 -O3 -arch=sm_90a -Xcompiler -fPIC -shared -lcuda -DB2M_PROD=1 -DB2M_HDW=1"
build() {  # tag V extra...
  tag=$1; V=$2; shift 2
  so=/tmp/b2wk_${tag}_v${V}.so
  nvcc $BASE -DB2M_VD=$V "$@" "$CU" -o "$so" \
    && echo "  built ${tag} V=${V}" || { echo "  FAIL ${tag} V=${V}"; exit 1; }
}
for V in 256 128; do
  build prod       $V
  build bf16       $V -DB2M_BF16=1
  build prod64     $V -DB2M_BVK=64
  build bf16tma    $V -DB2M_BVK=64 -DB2M_BF16=1 -DB2M_TMA=1
  if [ "$V" = 256 ]; then
    build bf16mma     $V -DB2M_BF16=1 -DB2M_BF16MMA=1
    build bf16mmatma  $V -DB2M_BVK=64 -DB2M_BF16=1 -DB2M_BF16MMA=1 -DB2M_TMA=1
  fi
done

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

prof() {  # shape V tag skip
  shape=$1; V=$2; tag=$3; skip=${4:-3}
  echo ""
  echo "################################################################################"
  echo "# b2prod_launch  shape=${shape}  V=${V}  tag=${tag}  launch_skip=${skip}"
  echo "################################################################################"
  B2KV_SO=/tmp/b2wk_${tag}_v${V}.so B2KV_V=$V B2KV_TAG=$tag \
    ncu -k "regex:b2mma_kernel" -c 1 --launch-skip "$skip" --metrics "$M" \
    python "$DIR/harness/ncu_b2_kv_driver.py" --shape "$shape" --reps 8 2>&1 \
      | sed -e '/^==PROF==/d' -e '/^==WARNING==/d'
}

echo ""
echo "=================== main table: V=256 x {prod,bf16,bf16mma,prod64,bf16tma,bf16mmatma} ==================="
for tag in prod bf16 bf16mma prod64 bf16tma bf16mmatma; do
  prof b8t2048h8 256 "$tag"
done

echo ""
echo "=================== V=128: {prod,bf16,prod64,bf16tma} ==================="
for tag in prod bf16 prod64 bf16tma; do
  prof b8t2048h8 128 "$tag"
done

echo ""
echo "=================== determinism: V=256 bf16tma, launch_skip in {0,1,5} ==================="
for SK in 0 1 5; do
  prof b8t2048h8 256 bf16tma "$SK"
done
