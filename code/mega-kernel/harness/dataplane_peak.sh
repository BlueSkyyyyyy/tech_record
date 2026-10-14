#!/usr/bin/env bash
# P73-1（§61）复现：把 §59/§60 的「megakernel worker 数据面」从调度器骨架里解耦成纯带宽
#   微基准，单独扫到 HBM 读峰值，并在峰值下复测「控制面描述符写是否争 HBM」。
#   相对 §60 的唯一新变量：(a) 数据面与调度器解耦；(b) 数据面载体新增 TMA bulk→SMEM
#   （与寄存器 __ldcs uint4 对跑）；(c) 争用用同核并发 writer 块扫写速率找转折点。
#   补掉 §60.9 明留的尾巴「只到峰值 ~85%（T=1024 单 block/SM = 50% 占用），真 100% 峰值下
#   的争用仍留一丝未测」。
# 目标卡 H100 / sm90a，DEV 选空闲 device（默认 3）。
set -u
DEV=${DEV:-3}
DIR=${HOME}/proj/tech_record/code/mega-kernel
SRC=$DIR/src/dataplane_peak.cu
BIN=/tmp/dataplane_peak
nvcc -O3 -arch=sm_90a "$SRC" -o "$BIN" || exit 1

echo "########## §61 数据面到真 HBM 峰值 × 控制面争用 ##########"
CUDA_VISIBLE_DEVICES="$DEV" timeout 2400 "$BIN" bench
