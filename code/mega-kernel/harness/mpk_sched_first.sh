#!/usr/bin/env bash
# P25-1 continuation (round 5): ROOT CAUSE of the device-side hang, found with
# cuda-gdb + a launch-order ablation.
#
# Evidence chain:
#   (a) cuda-gdb attached to the *hung* process shows only ONE resident kernel
#       -- worker_kernel -- spinning in its wait-for-task loop
#       (persistent_kernel.cuh:1016 -> mpk_atoms.cuh:52 ld.acquire.gpu.u64).
#       scheduler_kernel is NOT resident, even at worker:scheduler = 4:4 where
#       the worker leaves 128 of 132 SMs free. So the hang is not "both
#       kernels spin" (the docs/09 section 10.4 reading from host stream
#       queries) but "the scheduler kernel never gets dispatched".
#   (b) launch-order ablation: swapping the two launches in
#       launch_persistent_kernel() so the scheduler is launched BEFORE the
#       worker makes the un-patched snapshot converge 4/4 (prewarm=none x3 &
#       prewarm=cast_1elem x1). The default order (worker first) still hangs.
#       => the worker's full persistent grid, launched first, starves the
#          scheduler kernel out of the device's kernel-dispatch pipeline.
#
# This script applies harness/mpk_sched_first.patch, runs the ablation, and
# restores the checkout (trap EXIT).
#
# Run inside kernel_lab:
#   bash harness/mpk_sched_first.sh
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export MIRAGE_HOME="${MIRAGE_HOME:-/ssd/mirage_mpk_build}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-7}"
PROBE="$HERE/mpk_race_probe.py"
PATCH="$HERE/mpk_sched_first.patch"
FILES=(include/mirage/persistent_kernel/persistent_kernel.cuh
       python/mirage/mpk/persistent_kernel.py)

restore() {
  git -C "$MIRAGE_HOME" checkout -- "${FILES[@]}" 2>/dev/null || true
}
trap restore EXIT

run() {  # run <label> <env-defines> <prewarm>
  local label="$1" defs="$2" pw="$3"
  echo "----- ${label} (MPK_EXTRA_DEFINES='${defs}' prewarm=${pw}) -----"
  if [ -n "$defs" ]; then export MPK_EXTRA_DEFINES="$defs"; else unset MPK_EXTRA_DEFINES; fi
  timeout 120 python3 -u "$PROBE" --device 0 --hang-timeout 25 --prewarm "$pw" 2>&1 \
      | grep -E "compiled in|RESULT|HANG"
  echo "exit=${PIPESTATUS[0]}"
}

echo "=== A) cuda-gdb already captured: harness/mpk_cudagdb{,_w4s4}.out.txt ==="
echo "=== B) launch-order ablation ==="

echo; echo "[B0] DEFAULT order (worker then scheduler), control"
run "default none" "" none

echo; echo "[B1] PATCH: scheduler launched BEFORE worker"
git -C "$MIRAGE_HOME" apply "$PATCH"
run "sched-first none #1" "-DMPK_DBG_SCHED_FIRST" none
run "sched-first none #2" "-DMPK_DBG_SCHED_FIRST" none
run "sched-first none #3" "-DMPK_DBG_SCHED_FIRST" none
run "sched-first cast_1elem" "-DMPK_DBG_SCHED_FIRST" cast_1elem
restore

echo; echo "=== done (source restored) ==="
