#!/usr/bin/env bash
# 在 kernel_lab 容器里用 ncu 剖析一个 .cu 源文件。
#
# 用法：
#   scripts/ncu.sh path/to/foo.cu --set full --kernel-name regex:foo [程序参数...]
#
# 说明：文件路径之后的所有参数都会原样交给 ncu；程序参数请用 `--` 显式分隔
# （ncu 自己的参数写在前面）。为方便起见，若没给 --kernel-name，默认抓所有 kernel。
# 环境变量：ARCH / NVCC_FLAGS / GPU 同 run.sh。
set -euo pipefail

SRC="${1:?usage: ncu.sh <file.cu> [ncu args...]}"; shift || true

# 按第一个 `--` 拆开：前面是 ncu 参数，后面是程序参数（放到可执行文件之后）。
NCU_OPTS=()
PROG_ARGS=()
seen_dd=0
for a in "$@"; do
  if [[ "$seen_dd" == 0 && "$a" == "--" ]]; then seen_dd=1; continue; fi
  if [[ "$seen_dd" == 1 ]]; then PROG_ARGS+=("$a"); else NCU_OPTS+=("$a"); fi
done

ABS="$(readlink -f "$SRC")"
DIR="$(dirname "$ABS")"
NAME="$(basename "$ABS" .cu)"
ARCH="${ARCH-sm_90}"
GPU="${GPU:-0}"
NVCC_FLAGS="${NVCC_FLAGS:-}"
ARCH_FLAG=""
[[ -n "$ARCH" ]] && ARCH_FLAG="-arch=$ARCH"

LAB="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/lab.sh"
"$LAB" up >/dev/null
LAB_NAME="${LAB_NAME:-kernel_lab}"

# 逐参数引用，避免 <、>、| 等被远端 shell 解释
NCU_ARGS=""
for a in "${NCU_OPTS[@]}"; do NCU_ARGS+=" $(printf '%q' "$a")"; done
APP_ARGS=""
for a in "${PROG_ARGS[@]}"; do APP_ARGS+=" $(printf '%q' "$a")"; done

docker exec -e CUDA_VISIBLE_DEVICES="$GPU" "$LAB_NAME" bash -lc \
  "cd '$DIR' && nvcc -O3 $ARCH_FLAG -lineinfo $NVCC_FLAGS '$NAME.cu' -o '$NAME.out' && \
   CUDA_VISIBLE_DEVICES=$GPU ncu$NCU_ARGS './$NAME.out'$APP_ARGS"
