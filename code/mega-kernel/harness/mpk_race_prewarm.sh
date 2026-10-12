#!/usr/bin/env bash
# Prewarm / grid-size sensitivity of the MPK megakernel hang.
#
# Runs harness/mpk_race_probe.py (single launch right after compile, exit
# 0=PASS / 1=WRONG / 124=HANG) for a matrix of "preceding GPU work" and for a
# matrix of worker/scheduler grid sizes, printing one line per run.
#
# Evidence that (a) the hang is not in the worker polling loop and (b) it is
# prevented by a preceding *row/whole-tensor reduction*.
#
# Run inside kernel_lab:
#   bash harness/mpk_race_prewarm.sh
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export MIRAGE_HOME="${MIRAGE_HOME:-/ssd/mirage_mpk_build}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-7}"
DEV=0
HT=20
TMO=110

run() {  # run <label> <extra probe args...>
  local label="$1"; shift
  local rc
  timeout "$TMO" python3 -u "$HERE/mpk_race_probe.py" --device "$DEV" \
      --hang-timeout "$HT" "$@" >/tmp/rp.log 2>&1
  rc=$?
  printf '  %-22s EXIT=%-3s %s%s\n' "$label" "$rc" \
      "$(grep -o 'RESULT=[A-Z]*' /tmp/rp.log | tail -1)" \
      "$(grep -o 'HANG after [0-9]*s' /tmp/rp.log | tail -1)"
}

echo "===== A. preceding-work (prewarm) sensitivity ====="
for pw in none mul clamp matmul softmax sumrows sumall mean rmsnorm; do
  run "prewarm=$pw" --prewarm "$pw"
done

echo
echo "===== B. worker:scheduler grid-size sensitivity (no prewarm) ====="
for cfg in 4:4 16:16 64:16 128:16; do
  w="${cfg%%:*}"; s="${cfg##*:}"
  run "grid=${w}:${s}" --workers "$w" --schedulers "$s"
done
