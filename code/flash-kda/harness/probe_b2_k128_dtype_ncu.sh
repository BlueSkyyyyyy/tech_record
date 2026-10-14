#!/usr/bin/env bash
# P7-91: ncu side -- pipe attribution of the six-segment production kernel's bf16-native entries
# as K moves in {64,128,256} (the "no pipe attribution" gap in docs/07 section 93.6).
#   prod       -DB2M_PROD -DB2M_HDW
#   bf16       + -DB2M_BF16
#   bf16mma    + -DB2M_BF16 -DB2M_BF16MMA
#   bf16_vdw   + -DB2M_BF16 -DB2M_BF16_VDW
# V=K at each K.  Shape b8t2048h8 (grid fills the SMs).
#
# NOTE: nvcc hangs under ncu in this container (P7-16) -> pre-build outside ncu.
# Usage (inside kernel_lab):
#   bash .../probe_b2_k128_dtype_ncu.sh > .../probe_b2_k128_dtype_ncu.out.txt 2>&1
set -uo pipefail

DIR=${HOME}/proj/tech_record/code/flash-kda
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-3}
cd "$DIR"
CU="$DIR/harness/b2_mma_probe.cu"

echo "== prebuild (outside ncu) =="
extra() {  # tag
  case $1 in
    prod)     echo "" ;;
    bf16)     echo "-DB2M_BF16=1" ;;
    bf16mma)  echo "-DB2M_BF16=1 -DB2M_BF16MMA=1" ;;
    bf16_vdw) echo "-DB2M_BF16=1 -DB2M_BF16_VDW=1" ;;
    *) echo "bad tag $1" >&2; exit 1 ;;
  esac
}
TAGS="prod bf16 bf16mma bf16_vdw"
for K in 64 128 256; do
  for tag in $TAGS; do
    so=/tmp/b2kdt_ncu_${tag}_k${K}_v${K}.so
    nvcc -std=c++17 -O3 -arch=sm_90a -Xcompiler -fPIC -shared -lcuda \
         -DB2M_PROD=1 -DB2M_HDW=1 -DB2M_KD=$K -DB2M_VD=$K $(extra "$tag") "$CU" -o "$so" \
      && echo "  built ${tag} K=${K}" || { echo "  FAIL ${tag} K=${K}"; exit 1; }
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

prof() {  # K tag
  K=$1; tag=$2
  echo ""
  echo "################################################################################"
  echo "# b2prod_launch  shape=b8t2048h8  K=V=${K}  tag=${tag}"
  echo "################################################################################"
  ncu -k "regex:b2mma_kernel" -c 1 --launch-skip 3 --metrics "$M" \
    python "$DIR/harness/ncu_b2_k128_dtype_driver.py" --shape b8t2048h8 --K "$K" --V "$K" \
      --tag "$tag" --so /tmp/b2kdt_ncu_${tag}_k${K}_v${K}.so --reps 6 2>&1 \
      | sed -e '/^==PROF==/d' -e '/^==WARNING==/d'
}

for K in 64 128 256; do
  for tag in $TAGS; do
    prof "$K" "$tag"
  done
done
