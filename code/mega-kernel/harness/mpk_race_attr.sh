#!/usr/bin/env bash
# Root-cause attribution for the MPK megakernel hang (P25-1).
#
# Runs harness/mpk_race_probe.py (single launch; exit 0=PASS / 1=WRONG /
# 124=HANG) over a matrix of *preceding GPU work* variants and a matrix of
# worker:scheduler grid sizes, printing one line per run.
#
# Discriminates between candidate causes of the intermittent hang:
#   * pure time (sleep), stream-sync (sync), allocation (alloc)
#   * same-dtype elementwise/copy (mul, copy, addrow, clamp), compute (matmul)
#   * device-wide reductions (sumall/sumrows/mean/rmsnorm/softmax)
#   * dtype-conversion kernels (cast, copyto32, to_half, to64) vs a no-op
#     .to() (to_same)
#
# Run inside kernel_lab:
#   bash harness/mpk_race_attr.sh
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export MIRAGE_HOME="${MIRAGE_HOME:-/ssd/mirage_mpk_build}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-7}"
DEV=0
HT=12
TMO=60

run() {  # run <label> <probe args...>
  local label="$1"; shift
  local rc
  timeout "$TMO" python3 -u "$HERE/mpk_race_probe.py" --device "$DEV" \
      --hang-timeout "$HT" "$@" >/tmp/ra.log 2>&1
  rc=$?
  printf '  %-14s EXIT=%-3s %s\n' "$label" "$rc" \
      "$(grep -o 'RESULT=[A-Z]*' /tmp/ra.log | tail -1)"
}

echo "===== A. preceding-work (prewarm) sensitivity ====="
for pw in none sync sleep copy mul clamp matmul \
          sumall sumrows mean softmax rmsnorm \
          alloc addrow cast copyto32 to_half to64 to_same; do
  run "prewarm=$pw" --prewarm "$pw"
done

echo
echo "===== B. worker:scheduler grid-size sensitivity (no prewarm) ====="
for cfg in 4:4 16:16 64:16 128:16; do
  w="${cfg%%:*}"; s="${cfg##*:}"
  run "grid=${w}:${s}" --workers "$w" --schedulers "$s"
done
