#!/usr/bin/env bash
# P145-1 复现：MPK SM100 UTCCP 前置的 scale warp-transpose 三实现等价性/映射（目标卡 H100 / sm90a）。
# 该重排本身是 arch-agnostic SMEM 代码，故可在 H100 上真跑；UTCCP/TMEM 指令本体是 SM100 专属，只建模不执行。
# 用 DEV 选择空闲 device（默认 3）。
set -u
DEV=${DEV:-3}
SRC=${HOME}/proj/tech_record/code/mega-kernel/src/tmem_utccp_layout.cu
BIN=/tmp/tmem_utccp_layout
nvcc -O3 -arch=sm_90a "$SRC" -o "$BIN" || exit 1
echo "\$ CUDA_VISIBLE_DEVICES=$DEV $BIN"
CUDA_VISIBLE_DEVICES="$DEV" "$BIN"
