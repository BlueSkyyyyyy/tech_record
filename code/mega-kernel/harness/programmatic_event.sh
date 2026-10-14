#!/usr/bin/env bash
# P32-1 复现：事件可见性的前移原语（目标卡 H100 / sm90）。
#   ProgrammaticEvent / LaunchCompletionEvent 能把「事件触发点」提前到 kernel 完成之前的哪一点？
# 用 DEV 选择空闲 device。programmatic / LCE 事件必须 cudaEventDisableTiming。
set -u
DEV=${DEV:-3}
SRC=${HOME}/proj/tech_record/code/mega-kernel/src/programmatic_event.cu
BIN=/tmp/programmatic_event
nvcc -O3 -arch=sm_90 "$SRC" -o "$BIN" || exit 1
echo "\$ CUDA_VISIBLE_DEVICES=$DEV $BIN 30 100000 100000 20000"
CUDA_VISIBLE_DEVICES="$DEV" "$BIN" 30 100000 100000 20000
