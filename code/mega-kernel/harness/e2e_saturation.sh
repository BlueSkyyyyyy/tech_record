#!/usr/bin/env bash
# P74-1（§62）复现：把 §60 端到端调度器骨架（producer 写环 + worker TMA 取 task + 折叠 +
#   HBM 数据面）推出「占用率旋钮」（多 block/SM + __launch_bounds__），看端到端能到 HBM
#   峰值的多少；并在**同一二进制**内用三档 mode（data-only / +fold / full）把 §60(85%)
#   与 §61(纯数据面 96%) 之间的**夹层成本**分解出来，最后在最优占用率下扫 PS 复测
#   「真描述符/环写是否争 HBM 带宽」。
#   补掉 §61.10 明留的尾巴 (a)「纯读微基准是能力上界、非端到端吞吐，差即夹层成本」
#   (b)「争用测试的 writer 是合成写、非真实描述符/环协议」。
# 目标卡 H100 / sm90a，DEV 选空闲 device（默认 3）。
set -u
DEV=${DEV:-3}
DIR=~/proj/tech_record/code/mega-kernel
SRC=$DIR/src/e2e_saturation.cu
BIN=/tmp/e2e_saturation
nvcc -O3 -arch=sm_90a "$SRC" -o "$BIN" || exit 1

echo "########## §62 端到端骨架占用率 × 夹层成本分解 ##########"
CUDA_VISIBLE_DEVICES="$DEV" timeout 2400 "$BIN" bench
