#!/usr/bin/env bash
# P56-1 复现：**取 task × 吃 HBM 的 task 执行（饱和侧）——数据面把 HBM 打满时控制面还能被藏住吗？**
# 补齐 §44.7（P55-1）明留的第一条尾巴：§44 在「数据面未打满 HBM」的区间（1 block/SM、~1.2–1.4 TB/s）
# 得到 `D=3≈max(fetch,exec)`；但「若执行把 HBM 打满，控制面字节是否转为可见争抢」未测。
# 本节把小 SMEM 环（K=8,D=2 ⇒ 5.6 KB）与放大 grid（= NSM·B, B=1..8）叠起来抬高占用率，让数据面读
# 逼近 HBM 峰值，再测同一组分解量（fetch_only / exec_only / D=1 / D=3）。
# 真实锚点（mirage 快照 f9eb70c）：worker 取 task = 全线程协作 16 B cp.async 进 SMEM 单缓冲
#   （persistent_kernel.cuh:1063-1078，cp_async_wait<0> :1076-1077）；论文 §5.3 pre-fetching
#   task descriptions（行 1364-1368）/ paged SMEM（行 1313-1325）；TaskDesc 352 B（论文 §6.1 行 1365）。
# 机制：G 个常驻 worker block；描述符 DW=88（352 B，≫L2）；数据面数组 NDESC·EDW 字（≫L2）。
#   取数载体 = TMA bulk + 硬件 mbarrier；D=1 单缓冲（= 快照），D>=2 预取；do_fetch=0 只跑执行。
# 四个 Section：A 正确性 / B 扫占用率（数据面饱和度）/ C 饱和态扫 EDW / D 饱和态 D=1 代价。
# 目标卡 H100 / sm90a，DEV 选空闲 device（默认 3）。
set -u
DEV=${DEV:-3}
SRC=~/proj/tech_record/code/mega-kernel/src/fetch_exec_saturate.cu
BIN=/tmp/fetch_exec_saturate
nvcc -O3 -arch=sm_90a "$SRC" -o "$BIN" || exit 1

CUDA_VISIBLE_DEVICES="$DEV" timeout 3600 "$BIN" bench
