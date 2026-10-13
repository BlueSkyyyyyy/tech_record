#!/usr/bin/env bash
# P57-1 复现：**取 task × 吃 HBM 的执行，跨「多 block/SM」维度：小 SMEM 环换高占用率，能同时喂满
# 数据面吗？** 补齐 §45.7（P56-1）明留的第一条尾巴「若要测多 block/SM 争 HBM 需另配小 SMEM 环，但
# 小 K 会让控制面 TMA 低效，两者不可兼得——留作后续」。
# 机制：G = NSM·B 个常驻 worker block，每 block T 线程；描述符 DW=88（352 B，≫L2）；数据面
#   NDESC·EDW 字（≫L2）。取数载体 = TMA bulk + 硬件 mbarrier；D=1 单缓冲（= 快照），D>=2 预取；
#   do_fetch=0 只跑执行。小 K ⇒ 小环 ⇒ 多 block/SM（占用率旋钮）。
# 四个 Section：A 正确性 / B 数据面饱和旋钮（总线程 vs 块内线程）/ C 控制面取数 K 敏感度 /
#   D 多 block + 小 K 下预取是否仍藏住控制面。
# 真实锚点（mirage 快照 f9eb70c）：worker 取 task = 全线程协作 16 B cp.async 进 SMEM 单缓冲
#   （persistent_kernel.cuh:1063-1078，cp_async_wait<0> :1076-1077）；论文 §5.3 pre-fetching
#   task descriptions（行 1364-1368）/ paged SMEM（行 1313-1325）；TaskDesc 352 B（论文 §6.1 行 1365）。
# 目标卡 H100 / sm90a，DEV 选空闲 device（默认 3）。
set -u
DEV=${DEV:-3}
SRC=~/proj/tech_record/code/mega-kernel/src/fetch_exec_multiblock.cu
BIN=/tmp/fetch_exec_multiblock
nvcc -O3 -arch=sm_90a "$SRC" -o "$BIN" || exit 1

CUDA_VISIBLE_DEVICES="$DEV" timeout 3600 "$BIN" bench
