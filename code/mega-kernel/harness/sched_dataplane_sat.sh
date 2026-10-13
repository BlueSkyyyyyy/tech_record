#!/usr/bin/env bash
# P72-1（§60）复现：补掉 §59.6/§59.9 明留的尾巴——「数据面只到 ~950 GB/s、未把 HBM 推到饱和，
#   故『描述符写与数据面读不争 HBM』只在远低于峰值区间成立」。唯一新变量 = 数据面的 ILP 展开
#   因子 U（每线程在飞 load 数）：§59 是单累加器依赖式读（1 个在飞 load），本节用 U 个独立累加器
#   把数据面推到 HBM 饱和，然后在饱和点重问「加 producer 是否拖慢数据面」。环协议/per-worker
#   拓扑/producer 轮转与 §58/§59 完全相同。整数累加可重排，sink 仍逐位对 host 参考。
# 目标卡 H100 / sm90a，DEV 选空闲 device（默认 3）。
set -u
DEV=${DEV:-3}
DIR=~/proj/tech_record/code/mega-kernel
SRC=$DIR/src/sched_dataplane_sat.cu
BIN=/tmp/sched_dataplane_sat
nvcc -O3 -arch=sm_90a "$SRC" -o "$BIN" || exit 1

echo "########## §60 数据面 HBM 饱和 × 控制面争用 ##########"
CUDA_VISIBLE_DEVICES="$DEV" timeout 2400 "$BIN" bench
