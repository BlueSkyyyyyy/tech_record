#!/usr/bin/env bash
# P77-1（§65）复现：把 §64 的 worker 侧复杂度阶梯补上**第三种取 task 载体**——`cp.async`
#   （MPK worker 的原生载体：16 B `cp.async` 单缓冲进 SMEM，`persistent_kernel.cuh:1063-1078`）。
#   补掉 §64.9 明留的尾巴「载体是寄存器读（plain）/ TMA，**未含 `cp.async`**（§42 的第三种载体）」：
#   在同一骨架、同一占用率（T=256,MINB=8,U=4）下扫 worker 环数 G，比较 plain / TMA / cp.async
#   三载体的「每 block prologue」，看 cp.async 的 setup 是否比 TMA 轻。
#   阶梯：0=plain 1=tma 2=tma+data 3=data-only 4=cpasync 5=cpasync+data。
# 目标卡 H100 / sm90a，DEV 选空闲 device（默认 3）。
set -u
DEV=${DEV:-3}
DIR=~/proj/tech_record/code/mega-kernel
SRC=$DIR/src/worker_carrier_cpasync.cu
BIN=/tmp/worker_carrier_cpasync
nvcc -O3 -arch=sm_90a "$SRC" -o "$BIN" || exit 1

echo "########## §65 worker 侧取 task 载体第三种：cp.async 加进 §64 复杂度阶梯 ##########"
CUDA_VISIBLE_DEVICES="$DEV" timeout 1800 "$BIN" bench
