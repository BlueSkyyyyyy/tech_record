#!/usr/bin/env bash
# P71-1（§59）复现：补掉 §58.10 明留的尾巴——「`CW` 是纯 ALU 延迟、不产生真实 HBM 流量」。
#   给 §58 的 worker 加一条**真实 HBM 数据面**（每 batch 流式读 `MD` 个 uint4，__ldcs 绕 L2），
#   把「PS*（喂饱 G 个 worker 所需 producer 数）-vs-task 重量」曲线从**纯 ALU** 推进到
#   **访存主导**，并测数据面与 producer 描述符写是否争 HBM。环协议/per-worker 拓扑/producer
#   轮转与 §58 完全相同，唯一新变量 = 访存重量 MD。骨架 = PS producer 块 + G worker 块单常驻。
# 目标卡 H100 / sm90a，DEV 选空闲 device（默认 3）。
set -u
DEV=${DEV:-3}
DIR=${HOME}/proj/tech_record/code/mega-kernel
SRC=$DIR/src/sched_perworker_ring_mem.cu
BIN=/tmp/sched_perworker_ring_mem
nvcc -O3 -arch=sm_90a "$SRC" -o "$BIN" || exit 1

echo "########## §59 真实 HBM 数据面下的 producer:worker 配比 ##########"
CUDA_VISIBLE_DEVICES="$DEV" timeout 2400 "$BIN" bench
