#!/usr/bin/env bash
# P78-1（§66）复现：把 §65 的 cp.async **单缓冲**升级成**深环（模板化 DCA 槽预取）**，
#   补掉 §65.9 明留的尾巴「cp.async 是单缓冲（快照形态），**未做 D 深环预取**（§42 已量预取增益
#   全在 D=1→2）」——在 §64/§65 的**端到端 worker 阶梯**（producer 环 + 数据面 + 背压，同一占用率
#   T=256,MINB=8,U=4）里对跑 plain / 单缓冲 cp.async / 深环 cp.async / TMA，量出深环的每 block
#   prologue 与单缓冲缝。
#   阶梯：0=plain 1=tma 2=tma+data 3=data-only 4=cpasync 5=cpasync+data 6=ca-ring 7=ca-ring+data。
# 目标卡 H100 / sm90a，DEV 选空闲 device（默认 3）。
set -u
DEV=${DEV:-3}
DIR=${HOME}/proj/tech_record/code/mega-kernel
SRC=$DIR/src/worker_carrier_cpasync_ring.cu
BIN=/tmp/worker_carrier_cpasync_ring
nvcc -O3 -arch=sm_90a "$SRC" -o "$BIN" || exit 1

echo "########## §66 worker 侧取 task 载体第四种：cp.async 深环（补 §65.9 尾巴） ##########"
CUDA_VISIBLE_DEVICES="$DEV" timeout 1800 "$BIN" bench
