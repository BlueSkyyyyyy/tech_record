#!/usr/bin/env bash
# P7-98: close the boundary left in docs/07 section 100.6 -- the section 99/100 write-path
# decomposition (L1 store sectors -> L2 write sectors + hit/miss -> DRAM write bytes) and its
# 判据 ("compare write volume by L1 global store sectors, NOT dram__bytes_write") were measured
# on the six dtypes/staging forms of the six-segment production kernel, but section 100.6 flags:
#   "-DB2M_INTRA（折叠）与 -DB2M_DAMMA、K≠V 交叉未复测".
# This probe retests those three crosses.
#
# All builds: `-DB2M_PROD -DB2M_HDW`.  Outputs are fp32 and (for the same operands) bit-identical
# across the INTRA/DAMMA knobs; the K!=V rows scale the output volume so the 判据 must track it.
#
#   == Part A: INTRA/DAMMA cross (K=V=128), V=256 ==
#     tag=prod          (fp32 baseline)
#     tag=damma         -DB2M_DAMMA=1
#     tag=intra         -DB2M_INTRA=1                       (b2prod_intra_launch, extra dAqk)
#     tag=intra_damma   -DB2M_INTRA=1 -DB2M_DAMMA=1
#     tag=bf16          -DB2M_BF16=1
#     tag=bf16_damma    -DB2M_BF16=1 -DB2M_DAMMA=1
#     tag=bf16_intra    -DB2M_BF16=1 -DB2M_INTRA=1
#   == Part B: K!=V cross (fp32 prod), V=128 and V=256 ==
#     (K,V) in {(64,128),(128,128),(256,128),(64,256),(256,256)}
#   == Part C: store-vectorization cross (`-DB2M_VECST`, P7-71) -- the 判据 value S_L1 is the
#     *store instruction flow*, so vectorizing the store halves S_L1 while U is unchanged; this
#     is the only known knob that changes S_L1 (dtype/staging/mma-ISA leave it bit-identical).
#
# nvcc hangs under ncu in this container (P7-16) -> pre-build outside ncu.  Idle GPU required.
# Usage (inside kernel_lab, absolute path):
#   bash /ssd/.../harness/probe_b2_write_cross_ncu.sh > /ssd/.../probe_b2_write_cross_ncu.out.txt 2>&1
set -uo pipefail

DIR=~/proj/tech_record/code/flash-kda
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-1}
cd "$DIR"
CU="$DIR/harness/b2_mma_probe.cu"

echo "== prebuild (outside ncu) =="
BASE="-std=c++17 -O3 -arch=sm_90a -Xcompiler -fPIC -shared -lcuda -DB2M_PROD=1 -DB2M_HDW=1"
build() {  # tag K V extra...
  tag=$1; K=$2; V=$3; shift 3
  so=/tmp/b2wc_${tag}_k${K}_v${V}.so
  nvcc $BASE -DB2M_KD=$K -DB2M_VD=$V "$@" "$CU" -o "$so" \
    && echo "  built ${tag} K=${K} V=${V}" || { echo "  FAIL ${tag} K=${K} V=${V}"; exit 1; }
}
# Part A: INTRA/DAMMA cross at K=V=128, V=256
for tag in prod damma intra intra_damma bf16 bf16_damma bf16_intra; do
  case $tag in
    prod)         extra="" ;;
    damma)        extra="-DB2M_DAMMA=1" ;;
    intra)        extra="-DB2M_INTRA=1" ;;
    intra_damma)  extra="-DB2M_INTRA=1 -DB2M_DAMMA=1" ;;
    bf16)         extra="-DB2M_BF16=1" ;;
    bf16_damma)   extra="-DB2M_BF16=1 -DB2M_DAMMA=1" ;;
    bf16_intra)   extra="-DB2M_BF16=1 -DB2M_INTRA=1" ;;
  esac
  build "$tag" 128 256 $extra
done
# Part B: K!=V cross (fp32 prod)
for KV in "64 128" "128 128" "256 128" "64 256" "256 256"; do
  set -- $KV; build prod "$1" "$2"
done
# Part C: store-vectorization cross
build prodvec 128 256 -DB2M_VECST=1
build prodvec 128 128 -DB2M_VECST=1

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
  B2KV_SO=/tmp/b2wc_${tag}_k${K}_v${V}.so B2KV_K=$K B2KV_V=$V B2KV_TAG=$tag \
    ncu -k "regex:b2mma_kernel" -c 1 --launch-skip "$skip" --metrics "$M" \
    python "$DIR/harness/ncu_b2_kv_driver.py" --shape "$shape" --reps 8 2>&1 \
      | sed -e '/^==PROF==/d' -e '/^==WARNING==/d'
}

echo ""
echo "=================== Part A: INTRA/DAMMA cross, K=V=128, V=256 ==================="
for tag in prod damma intra intra_damma bf16 bf16_damma bf16_intra; do
  prof b8t2048h8 128 256 "$tag"
done

echo ""
echo "=================== Part B: K!=V cross (fp32 prod) ==================="
for KV in "64 128" "128 128" "256 128" "64 256" "256 256"; do
  set -- $KV; prof b8t2048h8 "$1" "$2" prod
done

echo ""
echo "=================== Part C: store-vectorization cross (-DB2M_VECST) ==================="
prof b8t2048h8 128 256 prod
prof b8t2048h8 128 256 prodvec
prof b8t2048h8 128 128 prod
prof b8t2048h8 128 128 prodvec
