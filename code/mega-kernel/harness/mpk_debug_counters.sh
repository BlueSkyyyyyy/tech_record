#!/usr/bin/env bash
# P28-1 (docs/09 section 13): device-side NUMERIC event-level evidence for the
# worker<->scheduler hang, the "candidate (1)" left open in docs/09 section 11.7
# ("with -lineinfo only, `print config.*` yields no event counters").
#
# Idea: rebuild the JIT-generated persistent kernel with `-G` (device debug
# info) by appending it through MPK_EXTRA_DEFINES (plumbing patch
# harness/mpk_debug_build.patch, which touches NO kernel code), then attach
# cuda-gdb to the already-hung process and print the *numeric* values of
#   config.all_event_counters[i], config.all_event_num_triggers[i],
#   config.worker_queue_last_ready_task_id[i], the scheduler queue cursors and
#   the worker's shared wait-loop cursor (next/last_task_pos, queue_idx).
#
# This is the same non-invasive attach method as harness/mpk_cudagdb.sh; the
# only difference is the -G build so cuda-gdb can resolve struct fields/types.
#
# Run inside kernel_lab:
#   bash harness/mpk_debug_counters.sh
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export MIRAGE_HOME="${MIRAGE_HOME:-/ssd/mirage_mpk_build}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-7}"
export MPK_EXTRA_DEFINES="${MPK_EXTRA_DEFINES:--G}"
# The -G build blocks the host thread *inside cudaLaunchKernel* (the scheduler
# launch never returns), so "LAUNCHED" is never printed; keep the JIT tempdir
# alive so cuda-gdb can map the launcher module's debug symbols.
export MPK_KEEP_TEMPDIR=1
PROBE="${HERE}/mpk_cudagdb_probe.py"
CMDS="${HERE}/mpk_debug_counters.cmds"
PATCH="${HERE}/mpk_debug_build.patch"
PYFILE=python/mirage/mpk/persistent_kernel.py
LOG=/tmp/mpk_debug_counters_probe.log
ALIVE="${ALIVE:-240}"
WORKERS="${WORKERS:-128}"
SCHEDULERS="${SCHEDULERS:-16}"

restore() { git -C "$MIRAGE_HOME" checkout -- "$PYFILE" 2>/dev/null || true; }
trap restore EXIT

echo "[sh] applying debug-build plumbing patch (MPK_EXTRA_DEFINES -> ${MPK_EXTRA_DEFINES})"
git -C "$MIRAGE_HOME" apply "$PATCH" || { echo "[sh] patch failed to apply"; exit 2; }

: > "$LOG"
python3 -u "$PROBE" --device 0 --alive-after "$ALIVE" --prewarm none \
    --workers "$WORKERS" --schedulers "$SCHEDULERS" \
    >"$LOG" 2>&1 &
PPID_PROBE=$!
echo "[sh] probe pid(bash child)=$PPID_PROBE; waiting for kernel-launch marker ..."

PID=""
# The -G build never reaches "[probe] LAUNCHED": the host thread blocks inside
# the (second) cudaLaunchKernel. So wait for the C-side "smem size:" print,
# which happens just before the two launches.
for _ in $(seq 1 300); do
  if grep -q "smem size:" "$LOG" 2>/dev/null; then
    PID=$(pgrep -f '^python3 -u .*mpk_cudagdb_probe' | head -1)
    [ -n "$PID" ] && break
  fi
  if ! kill -0 "$PPID_PROBE" 2>/dev/null; then
    echo "[sh] probe exited early:"; cat "$LOG"; exit 3
  fi
  sleep 1
done

if [ -z "$PID" ]; then
  echo "[sh] no kernel-launch marker within timeout:"; cat "$LOG"; kill "$PPID_PROBE" 2>/dev/null; exit 4
fi

echo "[sh] probe launched (python pid=$PID). tail of log:"
tail -6 "$LOG"
sleep 6   # let the worker kernel settle into its spin loop

echo "==================== cuda-gdb attach (-G build) ===================="
GDB_ARGS=()
while IFS= read -r line; do
  [ -z "$line" ] && continue
  GDB_ARGS+=(-ex "$line")
done < "$CMDS"
cuda-gdb -q -p "$PID" -batch "${GDB_ARGS[@]}" 2>&1
echo "==================== cuda-gdb done ======================"

kill "$PID" 2>/dev/null
kill "$PPID_PROBE" 2>/dev/null
wait 2>/dev/null
echo "[sh] probe killed"
