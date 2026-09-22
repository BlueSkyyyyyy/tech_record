#!/usr/bin/env bash
# 在 kernel_lab 容器里编译并运行一个 .cu。用法：scripts/run.sh src/fp16/foo.cu [程序参数...]
# 环境变量：ARCH=sm_90  GPU=0  NVCC_FLAGS="..."  INCLUDES="-I/path"
set -euo pipefail

SRC="${1:?usage: run.sh <file.cu> [args...]}"; shift || true
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

docker exec -e CUDA_VISIBLE_DEVICES="$GPU" "$LAB_NAME" bash -lc \
  "cd '$DIR' && nvcc -O3 -arch=$ARCH -lineinfo $INCLUDES $NVCC_FLAGS '$NAME.cu' -o '$NAME.out' && \
   CUDA_VISIBLE_DEVICES=$GPU ./'$NAME.out' $*"
