#!/usr/bin/env bash
# P68-1（§57）复现：把「每 worker 独立环」（share factor s=G/P=1）这条 §55.7/§56.7
#   反复点名、却一直没实现的「出路」补齐，对照共享前缀环（s>1）——
#   D 是否不是自由性能旋钮，而是被「每个环被几个 worker 共享」这一拓扑锁死的。
#   骨架 = P producer 块 + G=s·P worker 块单常驻 kernel；worker 用 TMA bulk load 取 batch。
# 目标卡 H100 / sm90a，DEV 选空闲 device（默认 3）。
set -u
DEV=${DEV:-3}
DIR=~/proj/tech_record/code/mega-kernel
SRC=$DIR/src/perworker_ring.cu
BIN=/tmp/perworker_ring
nvcc -O3 -arch=sm_90a "$SRC" -o "$BIN" || exit 1

echo "########## §57 每 worker 独立环 vs 共享前缀环：D 被环拓扑锁死 ##########"
CUDA_VISIBLE_DEVICES="$DEV" timeout 2400 "$BIN" bench
