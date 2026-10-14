#!/usr/bin/env bash
# P7-31: ncu the two halves of the B1->B2 `dh` round-trip (docs/07 §34).
#
#   * K2-shaped TMA B1 (b1_tma_probe.cu -DB1T_ETMA) base vs -DB1T_NODH
#   * Triton B2 (`wy_dqkg`) base vs build_kernel(nodh=True)
#
# The CUPTI ablation (check_bwd_dh_roundtrip.py) says removing B1's `dh` store is worth
# ~33-36% of B1 (and up to 3.6x at BT=32 NS=2), while removing B2's `dh` load is only
# ~5-8%.  ncu's DRAM byte counters decide whether the B1 delta is *bandwidth* (write
# bytes drop by exactly the dh size) or an *issue/scheduling* effect (bytes barely move).
#
# Prebuild the .so outside ncu (run the driver once without --no-build), then:
#   bash .../harness/probe_dh_roundtrip_ncu.sh > .../probe_dh_roundtrip_ncu.out.txt 2>&1
set -uo pipefail

DIR=${HOME}/proj/tech_record/code/flash-kda
DRIVER=$DIR/harness/ncu_dh_roundtrip_driver.py
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-1}
cd "$DIR"

SECTIONS="--section SpeedOfLight --section Occupancy --section SchedulerStats \
--section WarpStateStats --section LaunchStats --section MemoryWorkloadAnalysis"

METRICS="dram__bytes_read.sum,dram__bytes_write.sum,\
lts__t_bytes.sum,\
sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active,\
smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio,\
smsp__average_warps_issue_stalled_short_scoreboard_per_issue_active.ratio,\
smsp__average_warps_issue_stalled_wait_per_issue_active.ratio,\
smsp__average_warps_issue_stalled_drain_per_issue_active.ratio,\
smsp__average_warps_issue_stalled_lg_throttle_per_issue_active.ratio,\
gpu__time_duration.sum"

tag() { sed -e '/^==PROF==/d' -e '/^==WARNING==/d'; }

# prebuild the B1 .so outside ncu
python "$DRIVER" --side b1 --B 8 --T 2048 --H 8 --bt 64 --ns 1 >/dev/null 2>&1
python "$DRIVER" --side b1 --nodh --B 8 --T 2048 --H 8 --bt 64 --ns 1 >/dev/null 2>&1
python "$DRIVER" --side b1 --B 2 --T 1024 --H 8 --bt 32 --ns 2 >/dev/null 2>&1
python "$DRIVER" --side b1 --nodh --B 2 --T 1024 --H 8 --bt 32 --ns 2 >/dev/null 2>&1

run() {  # label side nodh B T H bt ns kernel_regex
  local label=$1 side=$2 nodh=$3 B=$4 T=$5 H=$6 bt=$7 ns=$8 k=$9
  echo ""
  echo "################################################################################"
  echo "# ${label}"
  echo "#   side=${side} nodh=${nodh} B=${B} T=${T} H=${H} bt=${bt} ns=${ns} kernel=${k}"
  echo "################################################################################"
  ncu -k "regex:${k}" -c 1 --launch-skip 3 $SECTIONS --metrics "$METRICS" \
      python "$DRIVER" --side "$side" $nodh --B "$B" --T "$T" --H "$H" --bt "$bt" \
      --ns "$ns" --no-build 2>&1 | tag | grep -vE '^DONE'
}

echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "ncu: $(ncu --version | grep -i 'version' | head -1)"

run "B1 ETMA base  (B8 T2048H8 BT=64 NS=1)"        b1 ""    8 2048 8 64 1 b1_tma_kernel
run "B1 ETMA nodh  (B8 T2048H8 BT=64 NS=1)"        b1 "--nodh" 8 2048 8 64 1 b1_tma_kernel
run "B1 ETMA base  (B2 T1024H8 BT=32 NS=2)"        b1 ""    2 1024 8 32 2 b1_tma_kernel
run "B1 ETMA nodh  (B2 T1024H8 BT=32 NS=2)"        b1 "--nodh" 2 1024 8 32 2 b1_tma_kernel
run "B2 wy_dqkg base (B8 T2048H8 BT=64)"           b2 ""    8 2048 8 64 1 _wy_dqkg_kernel
run "B2 wy_dqkg nodh (B8 T2048H8 BT=64)"           b2 "--nodh" 8 2048 8 64 1 _wy_dqkg_kernel

echo ""
echo "DONE"
