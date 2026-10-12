#!/usr/bin/env bash
# P26-1: end-to-end latency of a *multi-operator* megakernel (rmsnorm ->
# linear(gate+up) -> silu_mul -> linear+residual) on the *clean*
# scheduler-first build.
#
# Follows docs/09 section 11: the clean workaround for the device-side hang is
# to launch the scheduler kernel BEFORE the worker kernel. This script applies
# harness/mpk_sched_first.patch (only swaps the two launches, no kernel-code
# change), sets MPK_EXTRA_DEFINES=-DMPK_DBG_SCHED_FIRST, runs
# harness/mpk_latency_multi.py, and restores the checkout (trap EXIT).
#
# Run inside kernel_lab:
#   bash harness/mpk_latency_multi.sh
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export MIRAGE_HOME="${MIRAGE_HOME:-/ssd/mirage_mpk_build}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-7}"
PATCH="$HERE/mpk_sched_first.patch"
PY="$HERE/mpk_latency_multi.py"
FILES=(include/mirage/persistent_kernel/persistent_kernel.cuh
       python/mirage/mpk/persistent_kernel.py)

restore() { git -C "$MIRAGE_HOME" checkout -- "${FILES[@]}" 2>/dev/null || true; }
trap restore EXIT

git -C "$MIRAGE_HOME" apply "$PATCH"
export MPK_EXTRA_DEFINES="-DMPK_DBG_SCHED_FIRST"

timeout 900 python3 -u "$PY" --device 0 --iters 200 --warmup 20 \
    --batch-list 1,4,8,16,32 --configs 128:16 \
    2>&1 | grep -E '^\[env\]|^\[MPK INIT\]|^ batch|^[[:space:]]+[0-9]+[[:space:]]|Traceback|Error|error:'
echo "exit=${PIPESTATUS[0]}"
restore
