#!/usr/bin/env bash
# P76-1（§64）复现：把 §62 端到端骨架里的 **worker 侧**放上一条复杂度阶梯（plain 寄存器读环 /
#   +TMA 取 task 进 SMEM / +HBM 数据面 / 仅数据面），在**同一二进制、同一占用率**
#   （T=256,MINB=8,U=4）下扫 worker 环数 G，把 §62 的「G 越大越慢」钉到**哪一层 worker 机制**。
#   补掉 §63.9 明留的尾巴「consumer 是 plain-load，不能直接证明 G 依赖来自 TMA 路径——需把
#   TMA+折叠+数据面按 §62 加回」。
# 目标卡 H100 / sm90a，DEV 选空闲 device（默认 3）。
set -u
DEV=${DEV:-3}
DIR=${HOME}/proj/tech_record/code/mega-kernel
SRC=$DIR/src/worker_side_attribution.cu
BIN=/tmp/worker_side_attribution
nvcc -O3 -arch=sm_90a "$SRC" -o "$BIN" || exit 1

echo "########## §64 worker 侧复杂度阶梯 × 环数 G 归因 ##########"
CUDA_VISIBLE_DEVICES="$DEV" timeout 1800 "$BIN" bench
