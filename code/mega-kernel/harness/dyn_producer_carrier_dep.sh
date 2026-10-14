#!/usr/bin/env bash
# P64-1 复现：把 §52（P63-1）的**写端握手归因**从**单一载体（TMA bulk store）**推进到**三种写载体**，
#   补掉 §52.8 明留的尾巴「换栈（如 v4/coop 协作写，见 §49/§50）会改变它，故本节比值不能外推到寄存器写载体」。
# 骨架 = §52 的六位正交编码（WA/BP/HD/REL/ACQ/FENCE）；唯一新变量 = 写载体：
#   carrier 0 = TMA bulk store（SMEM 暂存 → cp.async.bulk；= §52）
#   carrier 1 = 寄存器 copy-out（SMEM 暂存 → 全 block uint4 load/store）
#   carrier 2 = 寄存器直写（不落 SMEM；每线程生成 uint4 直接写环 = §49/§50 v4 形态）
# 目标卡 H100 / sm90a，DEV 选空闲 device（默认 3）。
set -u
DEV=${DEV:-3}
DIR=${HOME}/proj/tech_record/code/mega-kernel
SRC=$DIR/src/dyn_producer_carrier_dep.cu
BIN=/tmp/dyn_producer_carrier_dep
nvcc -O3 -arch=sm_90a "$SRC" -o "$BIN" || exit 1

echo "########## §cbench：写端握手归因 × 三载体 ##########"
CUDA_VISIBLE_DEVICES="$DEV" timeout 3600 "$BIN" cbench
