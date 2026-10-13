#!/usr/bin/env bash
# P7-96: close the section 87.5 "待验证" -- the bf16 build's `dram__bytes_write` is *higher* than
# fp32 at V=256 (+35 MB: 590.7 vs 555.7) even though every output tensor is fp32 and bit-identical.
# Hypotheses: (H1) run-to-run noise, (H2) 2-block/SM L2 dirty writeback, (H3) more store traffic.
# This probe decomposes the write path (L1 store sectors -> L2 write sectors + hit/miss -> DRAM write
# bytes) and adds a controlled occlusion knob `-DB2M_SMEM_SLACK` that reserves *unused* dynamic smem
# (no instruction changes) to force the bf16 build back to 1 block/SM.  It also sweeps
# `--launch-skip` (0/1/5) to test determinism vs warm/cold-L2 window.
#
# nvcc hangs under ncu in this container (P7-16) -> pre-build outside ncu.  Idle GPU required.
# Usage (inside kernel_lab, absolute path):
#   bash /ssd/.../harness/probe_b2_write_ncu.sh > /ssd/.../probe_b2_write_ncu.out.txt 2>&1
set -uo pipefail

DIR=~/proj/tech_record/code/flash-kda
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-1}
cd "$DIR"
CU="$DIR/harness/b2_mma_probe.cu"

echo "== prebuild (outside ncu) =="
build() {  # tag V extra...
  tag=$1; V=$2; shift 2
  so=/tmp/b2w_${tag}_v${V}.so
  nvcc -std=c++17 -O3 -arch=sm_90a -Xcompiler -fPIC -shared -lcuda \
       -DB2M_PROD=1 -DB2M_HDW=1 -DB2M_VD=$V "$@" "$CU" -o "$so" \
    && echo "  built ${tag} V=${V}" || { echo "  FAIL ${tag} V=${V}"; exit 1; }
}
# `-DB2M_SMEM_SLACK` reserves unused dynamic smem; 40960 B pushes the bf16 V=256 build from
# ~90 KB (2 blocks/SM) past the ~113.5 KB half-SM threshold to 1 block/SM, same instructions.
for V in 128 256; do
  build prod        $V
  build bf16        $V -DB2M_BF16=1
  build bf16slack   $V -DB2M_BF16=1 -DB2M_SMEM_SLACK=40960
done

M="dram__bytes_read.sum,dram__bytes_write.sum,dram__sectors_write.sum,dram__sectors_read.sum,\
lts__t_bytes.sum,lts__t_sectors_op_read.sum,lts__t_sectors_op_write.sum,\
lts__t_sectors_op_write_lookup_hit.sum,lts__t_sectors_op_write_lookup_miss.sum,\
lts__t_sectors_srcunit_tex_op_read.sum,\
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
  B2KV_SO=/tmp/b2w_${tag}_v${V}.so B2KV_V=$V B2KV_TAG=$tag \
    ncu -k "regex:b2mma_kernel" -c 1 --launch-skip "$skip" --metrics "$M" \
    python "$DIR/harness/ncu_b2_kv_driver.py" --shape "$shape" --reps 8 2>&1 \
      | sed -e '/^==PROF==/d' -e '/^==WARNING==/d'
}

echo ""
echo "=================== main table: V in {128,256} x {prod,bf16,bf16slack} ==================="
for V in 256 128; do
  for tag in prod bf16 bf16slack; do
    prof b8t2048h8 "$V" "$tag"
  done
done

echo ""
echo "=================== determinism: V=256, launch_skip in {0,1,5} ==================="
for tag in prod bf16; do
  for SK in 0 1 5; do
    prof b8t2048h8 256 "$tag" "$SK"
  done
done
