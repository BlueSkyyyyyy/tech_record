#!/usr/bin/env bash
# P37-1 复现：device graph launch 作为「in-kernel 分发」原语（目标卡 H100 / sm90a）。
#   设备侧图启动能不能替代 host 派发 / 单 kernel 内循环？
#   与 P4-1（host graph launch）、P36-1（in-kernel 循环 / 条件节点）对照。
# 编译：nvcc -O2 -arch=sm_90a -rdc=true ... -lcudadevrt（device 侧 runtime 需 rdc）。
# 用 DEV 选择空闲 device。
set -u
DEV=${DEV:-3}
SRC=${HOME}/proj/tech_record/code/mega-kernel/src/device_graph.cu
BIN=/tmp/device_graph
nvcc -O2 -arch=sm_90a -rdc=true "$SRC" -o "$BIN" -lcudadevrt || exit 1
echo "\$ CUDA_VISIBLE_DEVICES=$DEV $BIN 20000 5"
CUDA_VISIBLE_DEVICES="$DEV" "$BIN" 20000 5
