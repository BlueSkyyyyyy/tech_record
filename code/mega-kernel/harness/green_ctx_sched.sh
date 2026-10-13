#!/usr/bin/env bash
# P38-1 复现：megakernel 不能独占整机时 —— 持久化内核对共存负载的干扰 & Green Contexts SM 分区隔离
#   目标卡 H100 / sm90, CUDA 13.2（cudaDevSmResourceSplit / cudaGreenCtxCreate）。
#   与 P10-1（网格/占用率）、P14-1（占用率代价）对照：把「整机占用」推进到「多租户共存」维度。
# 用 DEV 选择空闲 device（默认 3）。
set -u
DEV=${DEV:-3}
MB=${MB:-256}
ROUNDS=${ROUNDS:-64}
SRC=~/proj/tech_record/code/mega-kernel/src/green_ctx_sched.cu
BIN=/tmp/green_ctx_sched
nvcc -O2 -arch=sm_90 "$SRC" -o "$BIN" || exit 1
echo "\$ CUDA_VISIBLE_DEVICES=$DEV $BIN $MB $ROUNDS"
CUDA_VISIBLE_DEVICES="$DEV" "$BIN" "$MB" "$ROUNDS"
