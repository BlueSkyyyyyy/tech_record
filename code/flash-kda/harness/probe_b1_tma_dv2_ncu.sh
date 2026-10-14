#!/usr/bin/env bash
# P7-33: ncu the B1 `dv2` TMA store (docs/07 §36) against the scalar / no-store variants.
# §35.5's remaining question: `dv2` is the *same* kind of per-warp token-major tile as
# `dh`, so does TMA also move it off the LSU path (drop `lg_throttle` / L2 write
# amplification) and how much does it add on top of the TMA `dh` store (P7-32)?
#
#   base    scalar dh + scalar dv2
#   tma     TMA dh  + scalar dv2      (P7-32)
#   tma2    TMA dh  + TMA dv2         (this round)
#   nodv2   scalar dh, no dv2         (upper bound for the dv2 store)
#   nodh    TMA? no: ETMA, no dh      (upper bound for the dh store)
#
# Prebuild the .so outside ncu (nvcc hangs under ncu), then:
#   bash .../harness/probe_b1_tma_dv2_ncu.sh > .../probe_b1_tma_dv2_ncu.out.txt 2>&1
set -uo pipefail

DIR=${HOME}/proj/tech_record/code/flash-kda
DRIVER=$DIR/harness/ncu_b1_tma_dv2_driver.py
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-1}
cd "$DIR"

SECTIONS=""

METRICS="dram__bytes_read.sum,dram__bytes_write.sum,\
lts__t_bytes.sum,\
sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active,\
smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio,\
smsp__average_warps_issue_stalled_short_scoreboard_per_issue_active.ratio,\
smsp__average_warps_issue_stalled_wait_per_issue_active.ratio,\
smsp__average_warps_issue_stalled_lg_throttle_per_issue_active.ratio,\
sm__throughput.avg.pct_of_peak_sustained_elapsed,\
gpu__time_duration.sum"

tag() { sed -e '/^==PROF==/d' -e '/^==WARNING==/d'; }

# prebuild all variant .so outside ncu
for v in base tma tma2 nodv2 nodh; do
  python "$DRIVER" --variant "$v" >/dev/null 2>&1
done

run() {  # label variant B T H bt ns
  local label=$1 variant=$2 B=$3 T=$4 H=$5 bt=$6 ns=$7
  echo ""
  echo "################################################################################"
  echo "# ${label}"
  echo "#   B=${B} T=${T} H=${H} bt=${bt} ns=${ns}"
  echo "################################################################################"
  timeout 420 ncu -k "regex:b1_tma_kernel" -c 1 --launch-skip 3 $SECTIONS --metrics "$METRICS" \
      python "$DRIVER" --variant "$variant" --B "$B" --T "$T" --H "$H" --bt "$bt" \
      --ns "$ns" --no-build 2>&1 | tag | grep -vE '^DONE'
}

echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "ncu: $(ncu --version | grep -i 'version' | head -1)"

for cfg in "8 2048 8 64 1" "2 1024 8 32 2"; do
  set -- $cfg
  run "base  scalar dh+dv2 (B$1 T$2H$3 BT=$4 NS=$5)" base  "$1" "$2" "$3" "$4" "$5"
  run "tma   TMA dh only  (B$1 T$2H$3 BT=$4 NS=$5)" tma   "$1" "$2" "$3" "$4" "$5"
  run "tma2  TMA dh+dv2   (B$1 T$2H$3 BT=$4 NS=$5)" tma2  "$1" "$2" "$3" "$4" "$5"
  run "nodv2 no dv2       (B$1 T$2H$3 BT=$4 NS=$5)" nodv2 "$1" "$2" "$3" "$4" "$5"
done

echo ""
echo "DONE"
