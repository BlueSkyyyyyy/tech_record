#!/usr/bin/env bash
# P7-32: ncu the TMA `dh` store variant (docs/07 §35) against the scalar store and the
# no-store ablation.  The question §34.4 left open: §34.1 localized the scalar store's
# cost to `lg_throttle` (+3x L2 write amplification); does a TMA store put it back on a
# fast path, and if so does `lg_throttle` actually drop?
#
#   base  -DB1T_ETMA              scalar per-thread `st.global`
#   nodh  -DB1T_ETMA -DB1T_NODH   no `dh` at all (the §34.1 upper bound)
#   tma   -DB1T_ETMA -DB1T_TMADH  `cp.async.bulk.tensor.5d` store
#
# Prebuild the .so outside ncu (nvcc hangs under ncu), then:
#   bash .../harness/probe_b1_tma_store_ncu.sh > .../probe_b1_tma_store_ncu.out.txt 2>&1
set -uo pipefail

DIR=${HOME}/proj/tech_record/code/flash-kda
DRIVER=$DIR/harness/ncu_dh_roundtrip_driver.py
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-1}
cd "$DIR"

SECTIONS=""   # metrics-only: the full section set replays many times and is flaky here

METRICS="dram__bytes_read.sum,dram__bytes_write.sum,\
lts__t_bytes.sum,\
sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active,\
smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio,\
smsp__average_warps_issue_stalled_short_scoreboard_per_issue_active.ratio,\
smsp__average_warps_issue_stalled_wait_per_issue_active.ratio,\
smsp__average_warps_issue_stalled_drain_per_issue_active.ratio,\
smsp__average_warps_issue_stalled_lg_throttle_per_issue_active.ratio,\
sm__throughput.avg.pct_of_peak_sustained_elapsed,\
gpu__time_duration.sum"

tag() { sed -e '/^==PROF==/d' -e '/^==WARNING==/d'; }

# prebuild the B1 .so outside ncu
for cfg in "8 2048 8 64 1" "2 1024 8 32 2"; do
  set -- $cfg
  python "$DRIVER" --side b1          --B "$1" --T "$2" --H "$3" --bt "$4" --ns "$5" >/dev/null 2>&1
  python "$DRIVER" --side b1 --nodh   --B "$1" --T "$2" --H "$3" --bt "$4" --ns "$5" >/dev/null 2>&1
  python "$DRIVER" --side b1 --tma    --B "$1" --T "$2" --H "$3" --bt "$4" --ns "$5" >/dev/null 2>&1
done

run() {  # label variant B T H bt ns
  local label=$1 variant=$2 B=$3 T=$4 H=$5 bt=$6 ns=$7
  echo ""
  echo "################################################################################"
  echo "# ${label}"
  echo "#   B=${B} T=${T} H=${H} bt=${bt} ns=${ns}"
  echo "################################################################################"
  ncu -k "regex:b1_tma_kernel" -c 1 --launch-skip 3 $SECTIONS --metrics "$METRICS" \
      python "$DRIVER" --side b1 $variant --B "$B" --T "$T" --H "$H" --bt "$bt" \
      --ns "$ns" --no-build 2>&1 | tag | grep -vE '^DONE'
}

echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "ncu: $(ncu --version | grep -i 'version' | head -1)"

for cfg in "8 2048 8 64 1" "2 1024 8 32 2"; do
  set -- $cfg
  run "B1 scalar (base)  (B$1 T$2H$3 BT=$4 NS=$5)" ""      "$1" "$2" "$3" "$4" "$5"
  run "B1 nodh          (B$1 T$2H$3 BT=$4 NS=$5)" "--nodh" "$1" "$2" "$3" "$4" "$5"
  run "B1 TMA  (tma)    (B$1 T$2H$3 BT=$4 NS=$5)" "--tma"  "$1" "$2" "$3" "$4" "$5"
done

echo ""
echo "DONE"
