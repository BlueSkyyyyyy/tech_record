#!/usr/bin/env bash
# 用 ncu 剖析一个 .cu。用法：scripts/ncu.sh src/fp16/foo.cu --set full --kernel-name regex:foo [-- 程序参数...]
# 环境变量同 run.sh。
set -euo pipefail

SRC="${1:?usage: ncu.sh <file.cu> [ncu args...]}"; shift || true
ABS="$(readlink -f "$SRC")"
DIR="$(dirname "$ABS")"
NAME="$(basename "$ABS" .cu)"
ARCH="${ARCH:-sm_90}"
GPU="${GPU:-0}"
NVCC_FLAGS="${NVCC_FLAGS:-}"
INCLUDES="${INCLUDES:-}"

LAB="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/lab.sh"
"$LAB" up >/dev/null
LAB_NAME="${LAB_NAME:-kernel_lab}"

NCU_ARGS=""
for a in "$@"; do NCU_ARGS+=" $(printf '%q' "$a")"; done

docker exec -e CUDA_VISIBLE_DEVICES="$GPU" "$LAB_NAME" bash -lc \
  "cd '$DIR' && nvcc -O3 -arch=$ARCH -lineinfo $INCLUDES $NVCC_FLAGS '$NAME.cu' -o '$NAME.out' && \
   CUDA_VISIBLE_DEVICES=$GPU ncu$NCU_ARGS './$NAME.out'"
