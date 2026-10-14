#!/usr/bin/env bash
# P54-1 复现：**worker「取 task」与「执行 task」的流水——描述符取数能否被 task 执行藏住？**
# 补齐 §39–§42（P50-1..P53-1）的整合缺口：那些小节全在「只取数、不执行 task」的口径下量载体/缓冲，
# 本节加入**真实 task 执行负载**（数据相关折叠 + R 轮串行 LCG 计算压舱），把「单缓冲串行(D=1，= 快照
#   cp_async_wait<0>)」与「预取(D≥2)」公平对照。
# 真实锚点（mirage 快照 f9eb70c）：worker 取 task = 全线程协作 16 B cp.async 进 SMEM 单缓冲
#   （persistent_kernel.cuh:1063-1078，cp_async_wait<0> :1076-1077）；论文 §5.3 pre-fetching
#   task descriptions（行 1364-1368）/ paged shared memory（行 1313-1325）；TaskDesc 352 B（论文 §6.1 行 1365）。
# 机制：G 个常驻 worker block；全局自描述描述符（DW=88 → 352 B，≫L2）。取数载体 = TMA bulk + 硬件
#   mbarrier；环形缓冲 = D 个槽（SMEM = D·K·352 B）。D=1 取完再执行；D≥2 执行 batch t 时预取 t+D。
# 四个 Section：A 正确性 / A2 交叉校验 / B 扫计算强度 R / C 扫环深 D / D 扫批量 K。
# 目标卡 H100 / sm90a，DEV 选空闲 device（默认 3）。
set -u
DEV=${DEV:-3}
SRC=${HOME}/proj/tech_record/code/mega-kernel/src/fetch_exec_overlap.cu
BIN=/tmp/fetch_exec_overlap
nvcc -O3 -arch=sm_90a "$SRC" -o "$BIN" || exit 1

CUDA_VISIBLE_DEVICES="$DEV" timeout 3600 "$BIN" bench
