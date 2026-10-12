#!/usr/bin/env bash
# P25-1 continuation: host-side poll of the two persistent kernels.
#
# Uses MPK_DBG_WAIT_TRACE (added by harness/mpk_launch_sync.patch at
# persistent_kernel.cuh): after launching worker_kernel + scheduler_kernel on
# their NonBlocking streams, the host non-blockingly queries cudaStreamQuery on
# both for ~2 s and prints which stream (if any) completes. This localizes the
# spinning kernel. Contrasts prewarm=none (HANG) with prewarm=cast_1elem (PASS).
#
# Run inside kernel_lab:
#   bash harness/mpk_wait_trace.sh
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export MIRAGE_HOME="${MIRAGE_HOME:-/ssd/mirage_mpk_build}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-7}"

for pw in none cast_1elem; do
  echo "===== prewarm=$pw (MPK_DBG_WAIT_TRACE) ====="
  MPK_EXTRA_DEFINES="-DMPK_DBG_WAIT_TRACE" timeout 90 python3 -u \
      "$HERE/mpk_race_probe.py" --device 0 --hang-timeout 15 \
      --prewarm "$pw" 2>&1 | grep -E "LSYNC|RESULT|HANG|worker kernel|sched" \
      | head -40
done
