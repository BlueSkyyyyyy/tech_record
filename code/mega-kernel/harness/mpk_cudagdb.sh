#!/usr/bin/env bash
# P25-1 continuation (candidate 1 of docs/09 section 10.7): device-side,
# event-level evidence for the worker<->scheduler deadlock, obtained by
# *attaching cuda-gdb to the already-hung process* instead of instrumenting
# the kernel (instrumentation perturbs the timing-sensitive race, see
# docs/09 sections 9.3/9.4).
#
# Steps:
#   1. launch the minimal RMSNorm megakernel on the PLAIN snapshot build with
#      prewarm=none -> it hangs, and the probe stays alive so we can attach;
#   2. wait for the "[probe] LAUNCHED pid=..." marker;
#   3. `cuda-gdb -p <pid> -batch -x mpk_cudagdb.cmds` -> dump running kernels,
#      focus each, print backtrace + PC + source line;
#   4. detach and kill the probe.
#
# Run inside kernel_lab:
#   bash harness/mpk_cudagdb.sh
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export MIRAGE_HOME="${MIRAGE_HOME:-/ssd/mirage_mpk_build}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-7}"
PROBE="${HERE}/mpk_cudagdb_probe.py"
CMDS="${HERE}/mpk_cudagdb.cmds"
LOG=/tmp/mpk_cudagdb_probe.log
ALIVE="${ALIVE:-240}"
WORKERS="${WORKERS:-128}"
SCHEDULERS="${SCHEDULERS:-16}"

: > "$LOG"
python3 -u "$PROBE" --device 0 --alive-after "$ALIVE" --prewarm none \
    --workers "$WORKERS" --schedulers "$SCHEDULERS" \
    >"$LOG" 2>&1 &
PPID_PROBE=$!
echo "[sh] probe pid(bash child)=$PPID_PROBE; waiting for LAUNCHED marker ..."

PID=""
for _ in $(seq 1 240); do
  if grep -q "LAUNCHED pid=" "$LOG" 2>/dev/null; then
    PID=$(grep -o "LAUNCHED pid=[0-9]*" "$LOG" | head -1 | cut -d= -f2)
    break
  fi
  if ! kill -0 "$PPID_PROBE" 2>/dev/null; then
    echo "[sh] probe exited early"; cat "$LOG"; exit 2
  fi
  sleep 1
done

if [ -z "$PID" ]; then
  echo "[sh] no LAUNCHED marker within timeout"; cat "$LOG"; kill "$PPID_PROBE" 2>/dev/null; exit 3
fi

echo "[sh] probe launched (python pid=$PID). tail of log:"
tail -5 "$LOG"
sleep 5   # let both persistent kernels settle into their spin loops

echo "==================== cuda-gdb attach ===================="
# Feed each command as a separate -ex so an error in one (e.g. focusing a
# kernel that does not exist) does not abort the remaining probes.
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
