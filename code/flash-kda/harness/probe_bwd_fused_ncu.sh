#!/usr/bin/env bash
# P7-27: ncu DRAM/latency attribution of the fused B1+B2 kernel vs the unfused pair.
# Run inside kernel_lab on an idle card:
#   CUDA_VISIBLE_DEVICES=1 bash /ssd/.../harness/probe_bwd_fused_ncu.sh
set -u
cd ${HOME}/proj/tech_record/code/flash-kda
export FLA_DISABLE_BACKEND_DISPATCH=1
B=${B:-8}; T=${T:-2048}; H=${H:-8}
MET="dram__bytes_read.sum,dram__bytes_write.sum,dram__throughput.avg.pct_of_peak_sustained_elapsed,gpu__time_duration.sum,sm__throughput.avg.pct_of_peak_sustained_elapsed,lts__throughput.avg.pct_of_peak_sustained_elapsed,sm__warps_active.avg.pct_of_peak_sustained_active,launch__registers_per_thread,launch__occupancy_limit_registers,launch__occupancy_limit_shared_mem"

echo "########## FUSED (B1+B2) ##########"
ncu -k regex:"_fused_b12_kernel" -c 1 --launch-skip 2 --metrics "$MET" \
    python harness/ncu_bwd_fused_driver.py --which fused --B "$B" --T "$T" --H "$H" \
    2>&1 | grep -Ev "^\s*$"

echo "########## PAIR (B1 dhu_bf16 + B2 wy_dqkg) ##########"
ncu -k regex:"_dhu_kernel|_wy_dqkg_kernel" -c 2 --launch-skip 4 --metrics "$MET" \
    python harness/ncu_bwd_fused_driver.py --which pair --B "$B" --T "$T" --H "$H" \
    2>&1 | grep -Ev "^\s*$"
