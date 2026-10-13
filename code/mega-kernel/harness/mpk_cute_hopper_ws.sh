#!/usr/bin/env bash
# Build + run MPK's Hopper CuTe weight-stationary GEMM standalone test
# (tests/runtime_python/cute/hopper/gemm.cu, entry `linear_mpk`) on the target
# card H100/sm90a. Output -> harness/mpk_cute_hopper_ws.out.txt
#
# Usage: code/mega-kernel/harness/mpk_cute_hopper_ws.sh [DEVICE]
set -uo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LAB="$DIR/../scripts/lab.sh"
DEV="${1:-4}"
OUT="$DIR/mpk_cute_hopper_ws.out.txt"

{
  echo "=== mpk cute/hopper weight-stationary GEMM standalone test ==="
  echo "date: $(date '+%F %T')  device: $DEV  nvcc: $(nvcc --version 2>/dev/null | tail -1)"
  echo
  "$LAB" exec "CUDA_VISIBLE_DEVICES=$DEV python $DIR/mpk_cute_hopper_ws.py" 2>&1 \
    | grep -E '^\[(info|correctness|perf|done)\]'
  echo
  echo "rc=${PIPESTATUS[0]}"
} | tee "$OUT"
