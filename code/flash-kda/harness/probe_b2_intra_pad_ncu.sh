#!/usr/bin/env bash
# P7-94: ncu side -- does the `intra` staging **stride pad** actually cut the shared-memory read
# pipe cost docs/07 §96.3 attributed the fold's ~26.6% K==128 tail to?  Profile `prod_intra`
# (`-DB2M_PROD -DB2M_HDW -DB2M_INTRA`) at K==128 for pad0 / q8 / q8s4 (see `B2M_INTRA_PAD` /
# `B2M_INTRA_SPAD` in b2_mma_probe.cu), shape b8t2048h8.
#
# NOTE: nvcc hangs under ncu in this container (P7-16) -> pre-build outside ncu.
# Usage (inside kernel_lab):
#   bash .../probe_b2_intra_pad_ncu.sh > .../probe_b2_intra_pad_ncu.out.txt 2>&1
set -uo pipefail

DIR=~/proj/tech_record/code/flash-kda
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-1}
cd "$DIR"
CU="$DIR/harness/b2_mma_probe.cu"
K=128

echo "== prebuild (outside ncu) =="
build() {
  local tag="$1"; shift
  nvcc -std=c++17 -O3 -arch=sm_90a -Xcompiler -fPIC -shared -lcuda \
       -DB2M_PROD=1 -DB2M_HDW=1 -DB2M_INTRA=1 -DB2M_KD=$K -DB2M_VD=$K "$@" "$CU" \
       -o /tmp/b2ip_ncu_${tag}.so && echo "  built ${tag}" \
    || { echo "  FAIL ${tag}"; exit 1; }
}
build pad0 -DB2M_INTRA_PAD=0 -DB2M_INTRA_SPAD=0
build q8   -DB2M_INTRA_PAD=8 -DB2M_INTRA_SPAD=0
build q8s4 -DB2M_INTRA_PAD=8 -DB2M_INTRA_SPAD=4

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

for tag in pad0 q8 q8s4; do
  echo ""
  echo "################################################################################"
  echo "# b2prod_intra_launch  shape=b8t2048h8  K=V=${K}  variant=${tag}"
  echo "################################################################################"
  ncu -k "regex:b2mma_kernel" -c 1 --launch-skip 3 --metrics "$M" \
    python "$DIR/harness/ncu_b2_k128_intra_driver.py" --shape b8t2048h8 --K "$K" --V "$K" \
      --which intra --so /tmp/b2ip_ncu_${tag}.so --reps 6 2>&1 \
    | sed -e '/^==PROF==/d' -e '/^==WARNING==/d'
done
