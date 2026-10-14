#!/usr/bin/env bash
# P66-1 复现：补掉 §54.8（P65-1）明留的尾巴「worker 取 task 环深固定 D=1：预取深度是 §41/§42 的
#   独立变量」+ §54 坑②「strided worker 预取 D>1 跨 producer 子环边界触发有界环死锁」。
#   把 §54 端到端流水里被钉死为 D=1 的「worker 取 task」维度放开，量：
#     (1) 预取深度 D 能否把 §43 每批串行缝藏住、抬高端到端吞吐；
#     (2) 分配形态 (strided bb=w+t*Gw / contig bb=w*chunk+t) 在有界环 + 动态生产者下谁是安全形态。
#   骨架 = §54 的 mode 21（动态生产者 gen + G worker + 可选数据面）+ 六位正交握手编码；
#   新增 `wassign`（0=strided, 1=contig）与 worker 侧守门狗。
# 目标卡 H100 / sm90a，DEV 选空闲 device（默认 3）。
set -u
DEV=${DEV:-3}
DIR=${HOME}/proj/tech_record/code/mega-kernel
SRC=$DIR/src/worker_prefetch_e2e.cu
BIN=/tmp/worker_prefetch_e2e
nvcc -O3 -arch=sm_90a "$SRC" -o "$BIN" || exit 1

echo "########## §wbench：worker 取 task 的预取深度 D × 分配形态（端到端） ##########"
CUDA_VISIBLE_DEVICES="$DEV" timeout 3600 "$BIN" wbench
