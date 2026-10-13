#!/usr/bin/env bash
# P146-1 复现：CUTLASS SM100 UMMA TMEM fragment 布局 trait（block-scaled scale-factor）的
# host 侧结构探测（目标卡 H100 / sm90a）。
# cute Layout 是 CUTE_HOST_DEVICE，故整套布局代数可在 host 上实例化/求值，无需 SM100 硬件；
# 这里用 MPK 快照钉死的 deps/cutlass（f3fde583）的真实头文件。
# cute 头不是 target-only，用 -arch=sm_90a 编译即可（不 launch 任何 kernel）。
set -u
DEV=${DEV:-3}
CUTLASS=${CUTLASS:-/ssd/mirage_mpk_build/deps/cutlass}
SRC=~/proj/tech_record/code/mega-kernel/src/sf_tmem_trait.cu
BIN=/tmp/sf_tmem_trait
nvcc -std=c++17 -O2 -arch=sm_90a -I"$CUTLASS/include" "$SRC" -o "$BIN" || exit 1
echo "\$ CUDA_VISIBLE_DEVICES=$DEV $BIN"
CUDA_VISIBLE_DEVICES="$DEV" "$BIN"
