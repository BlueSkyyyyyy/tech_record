#!/usr/bin/env bash
# P67-1（§56）复现：把 §55（P66-1）的「共享前缀 tail」环协议换成「每槽独立回收/就绪」
#   （per-slot reclaim/ready），在同一骨架 / 同一工作 / 同一分配形态上对跑，补掉 §55.8 明留的
#   两条尾巴：
#     (1)「contig 的静默错是『共享前缀 tail 协议』的性质，不是『连续分配』固有的——换 per-slot
#         回收协议后连续分配可正确」；
#     (2)「死锁/错误的具体边界未做闭式扫描」。
#   回答：per-slot 回收是否让 contig 正确、让 D>1 可用；prefix 下 strided+D>1 的死锁边界是否有
#   闭式；两协议开销差多少。
#   骨架 = P producer 块 + G worker 块单常驻 kernel；worker 用 TMA bulk load 取 batch。
# 目标卡 H100 / sm90a，DEV 选空闲 device（默认 3）。
set -u
DEV=${DEV:-3}
DIR=${HOME}/proj/tech_record/code/mega-kernel
SRC=$DIR/src/perslot_ring.cu
BIN=/tmp/perslot_ring
nvcc -O3 -arch=sm_90a "$SRC" -o "$BIN" || exit 1

echo "########## §56 per-slot 回收 vs 共享前缀：有界环 × 分配形态 × 预取深度 D ##########"
CUDA_VISIBLE_DEVICES="$DEV" timeout 2400 "$BIN" bench
