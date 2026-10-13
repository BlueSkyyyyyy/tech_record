#!/usr/bin/env bash
# P69-1（§58）复现：补掉 §57.8 明留的头号尾巴——真实 MPK 形态「**少 producer（scheduler）
#   × 多 per-worker 环**」。§57 的 `pair` 靠 P==G 拿到「一消费者一环」、grid=2G（producer
#   侧随 G 膨胀）；本节把「环所有权」与 block 解耦：PS 个 producer 各拥有连续一段 worker 环、
#   段内轮转填，每条环仍单生产者单消费者（前缀精确）、grid=PS+G。PS==G 退化为 §57 的 pair。
#   骨架 = PS producer 块 + G worker 块单常驻 kernel；worker 用 TMA bulk load 取 batch。
# 目标卡 H100 / sm90a，DEV 选空闲 device（默认 3）。
set -u
DEV=${DEV:-3}
DIR=~/proj/tech_record/code/mega-kernel
SRC=$DIR/src/sched_perworker_ring.cu
BIN=/tmp/sched_perworker_ring
nvcc -O3 -arch=sm_90a "$SRC" -o "$BIN" || exit 1

echo "########## §58 少 producer(scheduler) × 多 per-worker 环 ##########"
CUDA_VISIBLE_DEVICES="$DEV" timeout 2400 "$BIN" bench
