#!/usr/bin/env bash
# P144-1 复现：MPK smem_tma 布局 vs 硬件 TMA swizzle 的一致性验证（目标卡 H100 / sm90a）。
# 用 DEV 选择空闲 device（默认 3）。
set -u
DEV=${DEV:-3}
SRC=~/proj/tech_record/code/mega-kernel/src/tma_smem_layout.cu
BIN=/tmp/tma_smem_layout
nvcc -O3 -arch=sm_90a "$SRC" -o "$BIN" -lcuda || exit 1
echo "\$ CUDA_VISIBLE_DEVICES=$DEV $BIN"
CUDA_VISIBLE_DEVICES="$DEV" "$BIN"
