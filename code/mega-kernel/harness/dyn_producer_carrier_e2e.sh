#!/usr/bin/env bash
# P65-1 复现：把 §52/§53（P63-1/P64-1）的**写端握手归因**从**纯写口径（mode 9，grid=P，无 worker）**
#   推进到**端到端（有 worker + 数据面）**，补掉 §53.9 明留的尾巴「纯写口径……带 worker 的端到端
#   归因未在本节跑（§52.8 另一条尾巴，留待后续）」。
# 骨架 = §52 的六位正交编码（WA/BP/HD/REL/ACQ/FENCE），新增 **mode 21 = 动态生产者(gen) + G worker
#   + 可选数据面**：worker 用 TMA bulk load 取回描述符并执行（控制面折叠 + uint4 流式数据面读 + R 轮压舱）。
#   端到端墙钟 = max(producer 写侧, worker 取+执行侧)；扫数据面 EDW 与 P，看写侧握手税是否被隐藏。
# 目标卡 H100 / sm90a，DEV 选空闲 device（默认 3）。
set -u
DEV=${DEV:-3}
DIR=${HOME}/proj/tech_record/code/mega-kernel
SRC=$DIR/src/dyn_producer_carrier_e2e.cu
BIN=/tmp/dyn_producer_carrier_e2e
nvcc -O3 -arch=sm_90a "$SRC" -o "$BIN" || exit 1

echo "########## §ebench：写端握手归因的端到端形态（动态生产者 + worker + 数据面） ##########"
CUDA_VISIBLE_DEVICES="$DEV" timeout 3600 "$BIN" ebench
