#!/usr/bin/env bash
# P25-1 continuation: does the dtype-conversion "fix" depend on the *torch
# default stream*, or is it a device-global effect?
#
# Runs harness/mpk_race_probe.py (single launch; exit 0=PASS / 1=WRONG /
# 124=HANG). The `*_s2` prewarm variants issue the cast on an *independent,
# non-default* torch stream and then sync only that stream, so the torch
# default stream never sees the cast kernel. Contrast:
#   * cast            -> cast on the default stream        (known PASS)
#   * none            -> nothing                            (known HANG)
#   * cast_s2         -> cast on a side stream, synced      (this run)
#   * cast_s2_nosync  -> cast on a side stream, not synced
#   * copyto32_s2     -> widening copy_ on a side stream
#   * none_s2         -> create+sync a side stream, no kernel
#
# If cast_s2 PASSes, the avoidance is a device-global effect (any
# dtype-conversion kernel anywhere on the device), not default-stream history.
# If cast_s2 HANGs, the key is that the cast ran on the torch default stream.
#
# Run inside kernel_lab:
#   bash harness/mpk_race_stream.sh
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export MIRAGE_HOME="${MIRAGE_HOME:-/ssd/mirage_mpk_build}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-7}"
DEV=0
HT=20
TMO=70

run() {  # run <label> <probe args...>
  local label="$1"; shift
  local rc
  timeout "$TMO" python3 -u "$HERE/mpk_race_probe.py" --device "$DEV" \
      --hang-timeout "$HT" "$@" >/tmp/rs.log 2>&1
  rc=$?
  printf '  %-18s EXIT=%-4s %s%s\n' "$label" "$rc" \
      "$(grep -o 'RESULT=[A-Z]*' /tmp/rs.log | tail -1)" \
      "$(grep -o 'HANG after [0-9]*s' /tmp/rs.log | tail -1)"
}

echo "===== stream provenance of the dtype-conversion prewarm ====="
for pw in none cast cast_s2 cast_s2_nosync copyto32_s2 none_s2 cast_s2_keepalloc; do
  run "prewarm=$pw" --prewarm "$pw"
done
