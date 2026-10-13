#!/usr/bin/env bash
# P55-1 复现：**取 task 与「吃 HBM 的」task 执行争带宽——描述符取数还能被藏住吗？**
# 补齐 §43.7（P54-1）明留的第一条尾巴：§43 的 exec 是纯 ALU 压舱（不吃 HBM），故 `D=3=max(fetch,exec)`。
# 本节把数据面 HBM 读加进 exec（每条 task 流式读 EDW 个字），让「取 task（控制面，源=HBM）」与
# 「执行 task（数据面=HBM）」共用同一条带宽。
# 真实锚点（mirage 快照 f9eb70c）：worker 取 task = 全线程协作 16 B cp.async 进 SMEM 单缓冲
#   （persistent_kernel.cuh:1063-1078，cp_async_wait<0> :1076-1077）；论文 §5.3 pre-fetching
#   task descriptions（行 1364-1368）/ paged shared memory（行 1313-1325）；TaskDesc 352 B（论文 §6.1 行 1365）。
# 机制：G 个常驻 worker block；描述符 DW=88（352 B，≫L2）；数据面数组 NDESC·EDW 字（≫L2）。
#   取数载体 = TMA bulk + 硬件 mbarrier；D=1 单缓冲（= 快照），D=3 预取；do_fetch=0 只跑执行。
# 四个 Section：A 正确性 / B 扫 EDW（内存受限度）/ C 扫 R（计算强度）/ D 取数加价。
# 目标卡 H100 / sm90a，DEV 选空闲 device（默认 3）。
set -u
DEV=${DEV:-3}
SRC=~/proj/tech_record/code/mega-kernel/src/fetch_exec_hbm.cu
BIN=/tmp/fetch_exec_hbm
nvcc -O3 -arch=sm_90a "$SRC" -o "$BIN" || exit 1

CUDA_VISIBLE_DEVICES="$DEV" timeout 3600 "$BIN" bench
