#!/usr/bin/env bash
# P7-99: close the two crosses left in docs/07 section 101.7 (P7-98):
#   (1) INTRA/DAMMA cross had only been measured at V=256 -- add V=128.
#   (2) the INTRA/DAMMA knobs had never been crossed with TMA / BF16MMA ("three-way").
# Same three-metric write-path decomposition as section 99/100/101:
#   L1 global store sectors (S_L1) -> L2 write sectors + hit/miss -> dram__bytes_write.
# 判据: S_L1 is the store *instruction flow*; it must be bit-identical across dtype/staging/
# mma-ISA/INTRA/DAMMA/K!=V and track the analytic output volume U (S_L1*32/U = 2 default,
# = 1 with `-DB2M_VECST`).
#
# All builds: `-DB2M_PROD -DB2M_HDW`, K=V=128 unless stated; BT=16.
#
#   == Part A2: INTRA/DAMMA cross at V=128 ==
#     tag=prod / damma / intra / intra_damma      (K=V=128, V=128)
#   == Part D: three-way INTRA/DAMMA x TMA/BF16MMA (K=V=128, V=256) ==
#     tag=tma              -DB2M_TMA
#     tag=damma_tma        -DB2M_DAMMA -DB2M_TMA
#     tag=intra_tma        -DB2M_INTRA -DB2M_TMA            (composable since P7-67 / sec.70)
#     tag=bf16mma          -DB2M_BF16 -DB2M_BF16MMA
#     tag=bf16mma_damma    -DB2M_BF16 -DB2M_BF16MMA -DB2M_DAMMA
#     tag=bf16mma_intra    -DB2M_BF16 -DB2M_BF16MMA -DB2M_INTRA
#     tag=bf16mma_intra_tma -DB2M_BF16 -DB2M_BF16MMA -DB2M_INTRA -DB2M_TMA
#   == Part E: does `-DB2M_VECST`'s S_L1 halving survive INTRA/TMA? ==
#     tag=intra_vecst / intra_tma_vecst          (K=V=128, V=256)
#
# nvcc hangs under ncu in this container (P7-16) -> pre-build outside ncu.  Idle GPU required.
# Usage (inside kernel_lab, absolute path):
#   bash /ssd/.../harness/probe_b2_write_cross2_ncu.sh > /ssd/.../probe_b2_write_cross2_ncu.out.txt 2>&1
set -uo pipefail

DIR=${HOME}/proj/tech_record/code/flash-kda
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-1}
cd "$DIR"
CU="$DIR/harness/b2_mma_probe.cu"

echo "== prebuild (outside ncu) =="
BASE="-std=c++17 -O3 -arch=sm_90a -Xcompiler -fPIC -shared -lcuda -DB2M_PROD=1 -DB2M_HDW=1"
build() {  # tag K V extra...
  tag=$1; K=$2; V=$3; shift 3
  so=/tmp/b2wc2_${tag}_k${K}_v${V}.so
  nvcc $BASE -DB2M_KD=$K -DB2M_VD=$V "$@" "$CU" -o "$so" 2>/dev/null \
    && echo "  built ${tag} K=${K} V=${V}" || { echo "  FAIL ${tag} K=${K} V=${V}"; exit 1; }
}
# Part A2: INTRA/DAMMA cross at V=128
for tag in prod damma intra intra_damma; do
  case $tag in
    prod)        extra="" ;;
    damma)       extra="-DB2M_DAMMA=1" ;;
    intra)       extra="-DB2M_INTRA=1" ;;
    intra_damma) extra="-DB2M_INTRA=1 -DB2M_DAMMA=1" ;;
  esac
  build "$tag" 128 128 $extra
done
# Part D: three-way INTRA/DAMMA x TMA/BF16MMA at V=256
for tag in tma damma_tma intra_tma bf16mma bf16mma_damma bf16mma_intra bf16mma_intra_tma; do
  case $tag in
    tma)                  extra="-DB2M_TMA=1" ;;
    damma_tma)            extra="-DB2M_DAMMA=1 -DB2M_TMA=1" ;;
    intra_tma)            extra="-DB2M_INTRA=1 -DB2M_TMA=1" ;;
    bf16mma)              extra="-DB2M_BF16=1 -DB2M_BF16MMA=1" ;;
    bf16mma_damma)        extra="-DB2M_BF16=1 -DB2M_BF16MMA=1 -DB2M_DAMMA=1" ;;
    bf16mma_intra)        extra="-DB2M_BF16=1 -DB2M_BF16MMA=1 -DB2M_INTRA=1" ;;
    bf16mma_intra_tma)    extra="-DB2M_BF16=1 -DB2M_BF16MMA=1 -DB2M_INTRA=1 -DB2M_TMA=1 -DB2M_BVK=64" ;;
  esac
  build "$tag" 128 256 $extra
done
# Part E: vecst x INTRA/TMA
build intra           128 256 -DB2M_INTRA=1   # SASS baseline at V=256 (not profiled)
build intra_vecst     128 256 -DB2M_INTRA=1 -DB2M_VECST=1
build intra_tma_vecst 128 256 -DB2M_INTRA=1 -DB2M_TMA=1 -DB2M_VECST=1

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
  B2KV_SO=/tmp/b2wc2_${tag}_k${K}_v${V}.so B2KV_K=$K B2KV_V=$V B2KV_TAG=$tag \
    ncu -k "regex:b2mma_kernel" -c 1 --launch-skip "$skip" --metrics "$M" \
    python "$DIR/harness/ncu_b2_kv_driver.py" --shape "$shape" --reps 6 2>&1 \
      | sed -e '/^==PROF==/d' -e '/^==WARNING==/d'
}

echo ""
echo "=================== Part A2: INTRA/DAMMA cross, K=V=128, V=128 ==================="
for tag in prod damma intra intra_damma; do
  prof b8t2048h8 128 128 "$tag"
done

echo ""
echo "=================== Part D: three-way INTRA/DAMMA x TMA/BF16MMA, K=V=128, V=256 ==================="
for tag in tma damma_tma intra_tma bf16mma bf16mma_damma bf16mma_intra bf16mma_intra_tma; do
  prof b8t2048h8 128 256 "$tag"
done

echo ""
echo "=================== Part E: vecst x INTRA/TMA, K=V=128, V=256 ==================="
for tag in intra intra_vecst intra_tma_vecst; do
  prof b8t2048h8 128 256 "$tag"
done

echo ""
echo "== SASS store widths (default vs vecst, intra/intra_tma) =="
for tag in intra intra_vecst intra_tma intra_tma_vecst; do
  cuobjdump -sass "/tmp/b2wc2_${tag}_k128_v256.so" 2>/dev/null \
    | sed -n '/b2mma_kernel/,/^$/p' > "/tmp/sass2_${tag}.s"
  printf "  %-16s STG.E(32b)=%s  STG.E.64=%s  LDGSTS=%s  UBLKCP=%s\n" "$tag" \
    "$(grep -c 'STG.E ' "/tmp/sass2_${tag}.s")" \
    "$(grep -c 'STG.E.64' "/tmp/sass2_${tag}.s")" \
    "$(grep -c 'LDGSTS' "/tmp/sass2_${tag}.s")" \
    "$(grep -c 'UBLKCP' "/tmp/sass2_${tag}.s")"
done

echo ""
echo "DONE"
