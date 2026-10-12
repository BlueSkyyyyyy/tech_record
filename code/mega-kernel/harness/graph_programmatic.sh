#!/usr/bin/env bash
# P35-1 复现：显式构图 + PROGRAMMATIC 依赖边（目标卡 H100 / sm90）。
#   图的 `cudaGraphEdgeData` programmatic 边能否把下游 kernel 的放行点前移到「上游完成之前」？
#   与 stream 上的 PDL（§21：不做）以及 capture 图（P19-1）对照。
# 编译：nvcc -O3 -arch=sm_90；用 DEV 选择空闲 device。
set -u
DEV=${DEV:-3}
SRC=~/proj/tech_record/code/mega-kernel/src/graph_programmatic.cu
BIN=/tmp/graph_programmatic
nvcc -O3 -arch=sm_90 "$SRC" -o "$BIN" || exit 1
echo "\$ CUDA_VISIBLE_DEVICES=$DEV $BIN 30 32 100000 100000 20000 0"
CUDA_VISIBLE_DEVICES="$DEV" "$BIN" 30 32 100000 100000 20000 0
