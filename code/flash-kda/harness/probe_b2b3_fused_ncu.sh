#!/usr/bin/env bash
# P7-30: ncu bound/traffic analysis for the *fused* B2+B3 kernel vs the unfused B2 (`wy_dqkg`)
# and B3 (`intra`) at the deployed CHUNK=16 granularity, fp32 operands (P7-26 §29.1).
# Checks the fusion's traffic claim: DRAM read/write bytes, L2 traffic, tensor density.
#
# Usage (inside kernel_lab):
#   bash .../probe_b2b3_fused_ncu.sh > .../probe_b2b3_fused_ncu.out.txt 2>&1

set -uo pipefail

DIR=${HOME}/proj/tech_record/code/flash-kda
DRIVER=$DIR/harness/ncu_b2b3_fused_driver.py
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-1}
cd "$DIR"

SECTIONS="--section SpeedOfLight --section Occupancy --section LaunchStats \
--section MemoryWorkloadAnalysis"

METRICS="sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active,\
sm__pipe_fma_cycles_active.avg.pct_of_peak_sustained_active,\
smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio,\
smsp__average_warps_issue_stalled_short_scoreboard_per_issue_active.ratio,\
smsp__average_warps_issue_stalled_wait_per_issue_active.ratio,\
l1tex__throughput.avg.pct_of_peak_sustained_elapsed,\
lts__throughput.avg.pct_of_peak_sustained_elapsed,\
dram__throughput.avg.pct_of_peak_sustained_elapsed,\
dram__bytes_read.sum,dram__bytes_write.sum,lts__t_bytes.sum,\
sm__throughput.avg.pct_of_peak_sustained_elapsed"

tag() { sed -e '/^==PROF==/d' -e '/^==WARNING==/d'; }

run() {  # label which B T H bt nw k
  local label=$1 which=$2 B=$3 T=$4 H=$5 bt=$6 nw=$7 k=$8
  echo ""
  echo "################################################################################"
  echo "# ${label}"
  echo "#   which=${which} B=${B} T=${T} H=${H} bt=${bt} nw=${nw}  kernel=${k}"
  echo "################################################################################"
  ncu -k "regex:${k}" -c 1 --launch-skip 3 $SECTIONS --metrics "$METRICS" \
      python "$DRIVER" --which "$which" --B "$B" --T "$T" --H "$H" --bt "$bt" --nw "$nw" \
      2>&1 | tag | grep -vE '^DONE'
}

echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "ncu: $(ncu --version | grep -i 'version' | head -1)"
echo "torch: $(python -c 'import torch; print(torch.__version__)' 2>/dev/null)"

run "B2 wy_dqkg fp32 @BT=16 (deployed, nw=4)"  b2    8 2048 8 16 4 _wy_dqkg_kernel
run "B3 intra  fp32 @BT=16 (nw=4)"             b3    8 2048 8 16 4 _intra_kernel_mma
run "FUSED B2+B3 fp32 @BT=16 (nw=8, BV=128)"   fused 8 2048 8 16 8 _fused_b2b3_kernel
run "FUSED B2+B3 fp32 small @BT=16 (T512 H4)"  fused 1 512  4 16 8 _fused_b2b3_kernel

echo ""
echo "DONE"
