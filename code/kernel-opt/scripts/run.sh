#!/usr/bin/env bash
# 在 kernel_lab 容器里编译并运行一个 .cu 源文件。
#
# 用法：
#   scripts/run.sh path/to/foo.cu [程序参数...]
#
# 环境变量：
#   ARCH=sm_90              # nvcc -arch
#   NVCC_FLAGS="-Xptxas -v" # 额外 nvcc 参数
#   GPU=0                   # CUDA_VISIBLE_DEVICES
set -euo pipefail

SRC="${1:?usage: run.sh <file.cu> [args...]}"; shift || true
ABS="$(readlink -f "$SRC")"
DIR="$(dirname "$ABS")"
NAME="$(basename "$ABS" .cu)"
ARCH="${ARCH-sm_90}"
GPU="${GPU:-0}"
NVCC_FLAGS="${NVCC_FLAGS:-}"
# 注意：CUDA 13 的 nvcc 不认 `-arch=sm_90a`（会退化成 sm_90），wgmma 需显式
# `-gencode=arch=compute_90a,code=sm_90a`。把 ARCH 设为空串即可只用 NVCC_FLAGS 里的 gencode。
ARCH_FLAG=""
[[ -n "$ARCH" ]] && ARCH_FLAG="-arch=$ARCH"

LAB="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/lab.sh"
"$LAB" up >/dev/null

LAB_NAME="${LAB_NAME:-kernel_lab}"
CMD="cd '$DIR' && nvcc -O3 $ARCH_FLAG -lineinfo $NVCC_FLAGS '$NAME.cu' -o '$NAME.out' && CUDA_VISIBLE_DEVICES=$GPU ./'$NAME.out' $*"
docker exec -e CUDA_VISIBLE_DEVICES="$GPU" "$LAB_NAME" bash -lc "$CMD"
