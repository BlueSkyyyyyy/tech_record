#!/usr/bin/env bash
# P36-1 复现：CUDA Graph 条件/循环节点（IF / WHILE）能否让一张图表达「数据相关 / 可变行程」控制流？
#   对照 MPK「每 batch size 一张静态图」的局限；四策略 eager / unroll / cwhile / kinternal。
# 编译：nvcc -O3 -arch=sm_90；用 DEV 选择空闲 device。
set -u
DEV=${DEV:-3}
SRC=~/proj/tech_record/code/mega-kernel/src/graph_conditional.cu
BIN=/tmp/graph_conditional
nvcc -O3 -arch=sm_90 "$SRC" -o "$BIN" || exit 1
echo "\$ CUDA_VISIBLE_DEVICES=$DEV $BIN 50 64 32 256 64"
CUDA_VISIBLE_DEVICES="$DEV" "$BIN" 50 64 32 256 64
