#!/usr/bin/env bash
# P25-1 continuation: is the hang fixed by forcing host-side ordering between
# the prepare_kernel (torch default stream) and the worker/scheduler kernels
# (fresh NonBlocking streams)?
#
# Uses harness/mpk_launch_sync.patch, which adds two env-gated macros at
# persistent_kernel.cuh:1968 (right after cudaEventRecord(prepare_done_event)):
#   MPK_DBG_LAUNCH_SYNC -> cudaStreamSynchronize(default_stream)
#   MPK_DBG_DEVICE_SYNC -> cudaDeviceSynchronize()
# The macros are injected via MPK_EXTRA_DEFINES (persistent_kernel.py:371).
#
# If MPK_DBG_LAUNCH_SYNC turns HANG into PASS with *no device-code change*,
# the race is host-side launch/handshake ordering on the NonBlocking streams.
#
# Run inside kernel_lab:
#   bash harness/mpk_launch_sync.sh
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export MIRAGE_HOME="${MIRAGE_HOME:-/ssd/mirage_mpk_build}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-7}"
DEV=0
HT=20
TMO=70

run() {  # run <label> <defines> <probe args...>
  local label="$1"; local defs="$2"; shift 2
  local rc
  MPK_EXTRA_DEFINES="$defs" timeout "$TMO" python3 -u \
      "$HERE/mpk_race_probe.py" --device "$DEV" --hang-timeout "$HT" "$@" \
      >/tmp/rl.log 2>&1
  rc=$?
  printf '  %-26s EXIT=%-4s %s%s\n' "$label" "$rc" \
      "$(grep -o 'RESULT=[A-Z]*' /tmp/rl.log | tail -1)" \
      "$(grep -o 'HANG after [0-9]*s' /tmp/rl.log | tail -1)"
}

echo "===== A. launcher stream-ordering macros (no prewarm) ====="
run "baseline(none)"                ""                      --prewarm none
run "LAUNCH_SYNC"                   "-DMPK_DBG_LAUNCH_SYNC" --prewarm none
run "DEVICE_SYNC"                   "-DMPK_DBG_DEVICE_SYNC" --prewarm none

echo
echo "===== B. control: cast prewarm with/without LAUNCH_SYNC ====="
run "cast(none)"                    ""                      --prewarm cast
run "cast + LAUNCH_SYNC"            "-DMPK_DBG_LAUNCH_SYNC" --prewarm cast
